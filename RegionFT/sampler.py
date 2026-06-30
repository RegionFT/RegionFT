from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from RegionFT.core import Region
from RegionFT.oracle import CounterfactualOracle


@dataclass
class SamplingResult:
    tests: int = 0
    violations: int = 0
    selected_region_counts: Dict[int, int] = field(default_factory=dict)
    selected_region_violations: Dict[int, int] = field(default_factory=dict)
    used_fallback_weights: bool = False

    @property
    def success_rate(self) -> float:
        return self.violations / float(self.tests) if self.tests else 0.0


class RegionSampler:
    """Generate tests with region weights based on size and estimated IDI rate."""

    def __init__(
        self,
        sampling_risk_focus_alpha: float = 1.0,
        sampling_risk_floor: float = 0.00001,
        batch_size: int = 512,
    ):
        self.sampling_risk_focus_alpha = float(sampling_risk_focus_alpha)
        if self.sampling_risk_focus_alpha < 0:
            raise ValueError("sampling_risk_focus_alpha must be non-negative")
        self.sampling_risk_floor = float(sampling_risk_floor)
        if self.sampling_risk_floor < 0:
            raise ValueError("sampling_risk_floor must be non-negative")
        self.batch_size = max(1, int(batch_size))

    def sample(
        self,
        regions: Sequence[Region],
        oracle: CounterfactualOracle,
        rng: random.Random,
        draws: Optional[int] = None,
        runtime: Optional[float] = None,
        deadline_cpu: Optional[float] = None,
    ) -> SamplingResult:
        if draws is None and runtime is None and deadline_cpu is None:
            raise ValueError("draws, runtime, or deadline_cpu must be provided")
        if not regions:
            return SamplingResult()

        weights, used_fallback = self.weights(regions)
        result = SamplingResult(used_fallback_weights=used_fallback)

        region_lows = np.array([[int(lo) for lo, _ in r.bounds] for r in regions], dtype=np.int64)
        region_spans = np.array(
            [[int(hi) - int(lo) + 1 for lo, hi in r.bounds] for r in regions], dtype=np.int64
        )
        region_ids = [r.region_id for r in regions]
        region_index = range(len(regions))
        n_features = region_lows.shape[1]
        rng_np = np.random.default_rng(rng.randrange(2 ** 63))

        started_at = time.process_time()
        verbose = logging.getLogger().isEnabledFor(logging.INFO)
        next_report = started_at + 2.0

        while True:
            if draws is not None and result.tests >= draws:
                break
            now_cpu = time.process_time()
            if runtime is not None and now_cpu - started_at >= runtime:
                break
            if deadline_cpu is not None and now_cpu >= deadline_cpu:
                break

            # Select regions according to q, then sample inputs uniformly within them.
            batch_count = self.batch_size
            if draws is not None:
                batch_count = min(batch_count, draws - result.tests)
            if batch_count <= 0:
                break
            idxs = np.fromiter(
                rng.choices(region_index, weights=weights, k=batch_count),
                dtype=np.int64,
                count=batch_count,
            )
            anchors = region_lows[idxs] + (
                rng_np.random((batch_count, n_features)) * region_spans[idxs]
            ).astype(np.int64)
            checks = oracle.check_many(anchors.tolist(), phase="generation")
            result.tests += batch_count
            violation_mask = np.fromiter(
                (check.is_violation for check in checks), dtype=bool, count=len(checks)
            )
            result.violations += int(violation_mask.sum())
            region_counts = np.bincount(idxs, minlength=len(regions))
            # Track realized per-region IDI rates for partition calibration.
            violation_counts = np.bincount(idxs[violation_mask], minlength=len(regions))
            for region_pos in np.nonzero(region_counts)[0]:
                region_id = region_ids[region_pos]
                result.selected_region_counts[region_id] = (
                    result.selected_region_counts.get(region_id, 0) + int(region_counts[region_pos])
                )
            for region_pos in np.nonzero(violation_counts)[0]:
                region_id = region_ids[region_pos]
                result.selected_region_violations[region_id] = (
                    result.selected_region_violations.get(region_id, 0) + int(violation_counts[region_pos])
                )

            if verbose and now_cpu >= next_report:
                logging.info(
                    "[RegionFT]   generation: tests=%d idi=%d | cpu=%.0fs",
                    result.tests,
                    result.violations,
                    now_cpu - started_at,
                )
                next_report = now_cpu + 2.0

        return result

    def weights(self, regions: Sequence[Region]) -> Tuple[List[float], bool]:
        """Compute w(R) from size, floored estimated IDI rate, and IDI-focus exponent alpha."""

        weights = [
            max(0.0, float(region.size))
            * (max(0.0, self.sampling_risk_floor, region.stats.ratio) ** self.sampling_risk_focus_alpha)
            for region in regions
        ]
        if sum(weights) > 0:
            return weights, False

        # If all computed weights are zero, fall back to size-proportional sampling.
        fallback = [max(0.0, float(region.size)) for region in regions]
        if sum(fallback) > 0:
            return fallback, True
        return [1.0 for _ in regions], True

    def probabilities(self, regions: Sequence[Region]) -> Tuple[List[float], bool]:
        weights, used_fallback = self.weights(regions)
        total = sum(weights)
        if total <= 0:
            return [0.0 for _ in regions], used_fallback
        return [weight / total for weight in weights], used_fallback
