from __future__ import annotations

import logging
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence, Tuple

from RegionFT.core import Bounds, Region, RegionStats, normalize_bounds, region_size
from RegionFT.oracle import CounterfactualOracle


@dataclass(frozen=True)
class ScoringSample:
    """A scoring sample labeled by whether it is an IDI."""

    x: List[int]
    is_violation: bool


@dataclass
class RegionDraft:
    """A region awaiting a split."""

    region_id: int
    bounds: Bounds
    depth: int
    parent_id: int
    split_history: List[str] = field(default_factory=list)


@dataclass
class RegionScore:
    bounds: Bounds
    size: int
    target_samples: int
    samples: List[ScoringSample]

    @property
    def actual_samples(self) -> int:
        return len(self.samples)

    @property
    def violations(self) -> int:
        return sum(1 for sample in self.samples if sample.is_violation)

    @property
    def ratio(self) -> float:
        return self.violations / float(self.actual_samples) if self.actual_samples else 0.0

    @property
    def gini(self) -> float:
        ratio = self.ratio
        return 2.0 * ratio * (1.0 - ratio)


@dataclass(frozen=True)
class SplitScore:
    attr_idx: int
    cut_value: int
    gain: float


class Partitioner:
    def partition(
        self,
        oracle: CounterfactualOracle,
        rng: random.Random,
        deadline_cpu: Optional[float] = None,
    ) -> List[Region]:
        raise NotImplementedError


class GiniPartitioner(Partitioner):
    """Build a Gini-guided partition from fresh, uniformly drawn scoring samples."""

    _EPSILON = 1e-12

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_per_region: int = 30,
        max_samples_per_region: int = 100,
        min_gain: float = 0.0,
        gini_weight_mode: str = "size",
        max_split_ratio: float = 1.0,
        batch_size: int = 512,
    ):
        if int(max_depth) < 0:
            raise ValueError("max_depth must be non-negative")
        if int(min_samples_per_region) < 0:
            raise ValueError("min_samples_per_region must be non-negative")
        if int(max_samples_per_region) < int(min_samples_per_region):
            raise ValueError("max_samples_per_region must be >= min_samples_per_region")
        self.max_depth = max(0, int(max_depth))
        self.min_samples_per_region = int(min_samples_per_region)
        self.max_samples_per_region = int(max_samples_per_region)
        self.min_gain = float(min_gain)
        self.gini_weight_mode = str(gini_weight_mode).lower()
        if self.gini_weight_mode not in ("size", "sample"):
            raise ValueError("gini_weight_mode must be 'size' or 'sample'")
        if self.gini_weight_mode == "sample":
            self._child_weight_fn = self._sample_child_weights
        else:
            self._child_weight_fn = self._size_child_weights
        self.max_split_ratio = float(max_split_ratio)
        if self.max_split_ratio < 0 or self.max_split_ratio > 1:
            raise ValueError("max_split_ratio must be between 0 and 1")
        self.batch_size = max(1, int(batch_size))
        self.root_size = 1

    def partition(
        self,
        oracle: CounterfactualOracle,
        rng: random.Random,
        deadline_cpu: Optional[float] = None,
    ) -> List[Region]:
        root_bounds = normalize_bounds(oracle.black_box_model.data_range)
        self.root_size = max(1, region_size(root_bounds))

        # Pi_active contains regions awaiting a split; FIFO gives breadth-first refinement.
        active: Deque[RegionDraft] = deque(
            [
                RegionDraft(
                    region_id=0,
                    bounds=root_bounds,
                    depth=0,
                    parent_id=-1,
                )
            ]
        )
        regions: List[Region] = []
        next_region_id = 1
        verbose = logging.getLogger().isEnabledFor(logging.INFO)
        partition_start_cpu = time.process_time()
        next_report = partition_start_cpu + 2.0

        while active:
            if verbose:
                now_cpu = time.process_time()
                if now_cpu >= next_report:
                    logging.info(
                        "[RegionFT]   partitioning: %d leaves, %d active | cpu=%.0fs",
                        len(regions),
                        len(active),
                        now_cpu - partition_start_cpu,
                    )
                    next_report = now_cpu + 2.0
            # Finalize unfinished regions at the deadline to preserve coverage of X.
            if self._deadline_reached(deadline_cpu):
                regions.extend(self._finalize_region(draft) for draft in active)
                break

            draft = active.popleft()
            score = self._score_region(draft, oracle, rng, deadline_cpu)

            best_split = None
            if (
                draft.depth < self.max_depth
                and score.ratio < self.max_split_ratio
                and not self._deadline_reached(deadline_cpu)
            ):
                # Select the non-protected attribute and cut with largest Gini gain.
                best_split = self._best_split(score, oracle.protected_attrs)

            if self._deadline_reached(deadline_cpu):
                # Avoid creating unscored children after the deadline.
                regions.append(self._finalize_region(draft, score, best_split))
                continue

            if best_split is None or best_split.gain <= self.min_gain + self._EPSILON:
                # A region without a qualifying split joins the final partition.
                regions.append(self._finalize_region(draft, score, best_split))
                continue

            left_bounds, right_bounds = self._split_bounds(
                draft.bounds,
                best_split.attr_idx,
                best_split.cut_value,
            )
            left = RegionDraft(
                region_id=next_region_id,
                parent_id=draft.region_id,
                bounds=left_bounds,
                depth=draft.depth + 1,
                split_history=draft.split_history
                + [f"x{best_split.attr_idx} <= {best_split.cut_value}"],
            )
            right = RegionDraft(
                region_id=next_region_id + 1,
                parent_id=draft.region_id,
                bounds=right_bounds,
                depth=draft.depth + 1,
                split_history=draft.split_history
                + [f"x{best_split.attr_idx} > {best_split.cut_value}"],
            )
            next_region_id += 2
            active.append(left)
            active.append(right)

        return regions

    def _score_region(
        self,
        draft: RegionDraft,
        oracle: CounterfactualOracle,
        rng: random.Random,
        deadline_cpu: Optional[float],
    ) -> RegionScore:
        """Draw uniform scoring samples and estimate the region's IDI rate."""

        size = region_size(draft.bounds)
        target_samples = self._target_samples(size)
        samples: List[ScoringSample] = []

        while len(samples) < target_samples:
            if self._deadline_reached(deadline_cpu):
                break
            batch_count = min(self.batch_size, target_samples - len(samples))
            anchors = [
                oracle.sample_from_bounds(draft.bounds, rng)
                for _ in range(batch_count)
            ]
            # The scoring label is whether the input is an IDI, not f(x).
            checks = oracle.check_many(anchors, phase="partition")
            if len(checks) != len(anchors):
                raise ValueError("CounterfactualOracle.check_many returned an unexpected result count")
            samples.extend(
                ScoringSample(check.anchor, check.is_violation)
                for check in checks
            )

        return RegionScore(
            bounds=draft.bounds,
            size=size,
            target_samples=target_samples,
            samples=samples,
        )

    def _target_samples(self, size: int) -> int:
        if self.max_samples_per_region == self.min_samples_per_region:
            return self.max_samples_per_region
        # Scale the scoring budget with sqrt(size(R) / size(X)).
        size_ratio = max(0.0, min(1.0, float(size) / float(self.root_size)))
        target = self.min_samples_per_region + (
            self.max_samples_per_region - self.min_samples_per_region
        ) * math.sqrt(size_ratio)
        return max(
            self.min_samples_per_region,
            min(self.max_samples_per_region, int(math.ceil(target))),
        )

    def _best_split(
        self,
        score: RegionScore,
        protected_attrs: Sequence[int],
    ) -> Optional[SplitScore]:
        if score.actual_samples < 2:
            return None

        best: Optional[SplitScore] = None
        protected_set = set(protected_attrs)
        for attr_idx, cut_value in self._candidate_splits(score, protected_set):
            split_score = self._score_split(score, attr_idx, cut_value)
            if split_score is None:
                continue
            if best is None or split_score.gain > best.gain:
                best = split_score
        return best

    def _candidate_splits(
        self,
        score: RegionScore,
        protected_attrs: set[int],
    ) -> List[Tuple[int, int]]:
        """Enumerate observed cuts for non-protected attributes."""

        candidates: List[Tuple[int, int]] = []
        for attr_idx, (lo, hi) in enumerate(score.bounds):
            if attr_idx in protected_attrs or int(lo) >= int(hi):
                continue
            values = sorted({sample.x[attr_idx] for sample in score.samples})
            for cut_value in values[:-1]:
                cut_value = int(cut_value)
                if int(lo) <= cut_value < int(hi):
                    candidates.append((attr_idx, cut_value))
        return candidates

    def _score_split(
        self,
        score: RegionScore,
        attr_idx: int,
        cut_value: int,
    ) -> Optional[SplitScore]:
        """Compute Gini gain using sample-count or child-region-size weights."""

        left_samples = [sample for sample in score.samples if sample.x[attr_idx] <= cut_value]
        right_samples = [sample for sample in score.samples if sample.x[attr_idx] > cut_value]
        if not left_samples or not right_samples:
            return None

        left_gini = self._gini(left_samples)
        right_gini = self._gini(right_samples)
        left_weight, right_weight = self._child_weight_fn(
            score,
            attr_idx,
            cut_value,
            len(left_samples),
            len(right_samples),
        )
        gain = score.gini
        gain -= left_weight * left_gini
        gain -= right_weight * right_gini
        return SplitScore(
            attr_idx=attr_idx,
            cut_value=cut_value,
            gain=gain,
        )

    def _sample_child_weights(
        self,
        score: RegionScore,
        attr_idx: int,
        cut_value: int,
        left_count: int,
        right_count: int,
    ) -> Tuple[float, float]:
        total = float(left_count + right_count)
        return left_count / total, right_count / total

    def _size_child_weights(
        self,
        score: RegionScore,
        attr_idx: int,
        cut_value: int,
        left_count: int,
        right_count: int,
    ) -> Tuple[float, float]:
        lo, hi = score.bounds[attr_idx]
        total_width = float(int(hi) - int(lo) + 1)
        left_width = float(int(cut_value) - int(lo) + 1)
        right_width = float(int(hi) - int(cut_value))
        return left_width / total_width, right_width / total_width

    def _finalize_region(
        self,
        draft: RegionDraft,
        score: Optional[RegionScore] = None,
        best_split: Optional[SplitScore] = None,
    ) -> Region:
        """Discard scoring samples and retain final-region statistics."""

        if score is None:
            score = RegionScore(
                bounds=draft.bounds,
                size=region_size(draft.bounds),
                target_samples=self._target_samples(region_size(draft.bounds)),
                samples=[],
            )
        return Region(
            region_id=draft.region_id,
            parent_id=draft.parent_id,
            bounds=draft.bounds,
            depth=draft.depth,
            size=score.size,
            stats=RegionStats(
                target_samples=score.target_samples,
                samples=score.actual_samples,
                violations=score.violations,
            ),
            split_history=draft.split_history,
            best_attr=None if best_split is None else best_split.attr_idx,
            best_cut=None if best_split is None else best_split.cut_value,
            best_gain=0.0 if best_split is None else best_split.gain,
        )

    def _split_bounds(
        self,
        bounds: Bounds,
        attr_idx: int,
        cut_value: int,
    ) -> Tuple[Bounds, Bounds]:
        left_bounds = list(bounds)
        right_bounds = list(bounds)
        lo, hi = bounds[attr_idx]
        left_bounds[attr_idx] = (int(lo), int(cut_value))
        right_bounds[attr_idx] = (int(cut_value) + 1, int(hi))
        return left_bounds, right_bounds

    def _gini(self, samples: Sequence[ScoringSample]) -> float:
        if not samples:
            return 0.0
        ratio = sum(1 for sample in samples if sample.is_violation) / float(len(samples))
        return 2.0 * ratio * (1.0 - ratio)

    def _deadline_reached(self, deadline_cpu: Optional[float]) -> bool:
        return deadline_cpu is not None and time.process_time() >= deadline_cpu


class RandomPartitioner(GiniPartitioner):
    """Build the RQ2 random partition with midpoint splits on non-protected attributes."""

    def partition(
        self,
        oracle: CounterfactualOracle,
        rng: random.Random,
        deadline_cpu: Optional[float] = None,
    ) -> List[Region]:
        root_bounds = normalize_bounds(oracle.black_box_model.data_range)
        self.root_size = max(1, region_size(root_bounds))
        protected = set(oracle.protected_attrs)

        # Build the random tree without oracle calls.
        leaves: List[RegionDraft] = []
        active: Deque[RegionDraft] = deque(
            [RegionDraft(region_id=0, bounds=root_bounds, depth=0, parent_id=-1)]
        )
        next_region_id = 1
        while active:
            draft = active.popleft()
            splittable = [
                idx
                for idx, (lo, hi) in enumerate(draft.bounds)
                if idx not in protected and int(lo) < int(hi)
            ]
            if draft.depth >= self.max_depth or not splittable:
                leaves.append(draft)
                continue
            attr = rng.choice(splittable)
            lo, hi = draft.bounds[attr]
            cut = (int(lo) + int(hi)) // 2
            left_bounds, right_bounds = self._split_bounds(draft.bounds, attr, cut)
            active.append(
                RegionDraft(
                    region_id=next_region_id,
                    parent_id=draft.region_id,
                    bounds=left_bounds,
                    depth=draft.depth + 1,
                    split_history=draft.split_history + [f"x{attr} <= {cut}"],
                )
            )
            active.append(
                RegionDraft(
                    region_id=next_region_id + 1,
                    parent_id=draft.region_id,
                    bounds=right_bounds,
                    depth=draft.depth + 1,
                    split_history=draft.split_history + [f"x{attr} > {cut}"],
                )
            )
            next_region_id += 2

        # Score each leaf to estimate its IDI rate for test generation.
        regions: List[Region] = []
        for draft in leaves:
            if self._deadline_reached(deadline_cpu):
                regions.append(self._finalize_region(draft))
                continue
            score = self._score_region(draft, oracle, rng, deadline_cpu)
            regions.append(self._finalize_region(draft, score, None))
        return regions


class PrecomputedPartitioner(Partitioner):
    """Reuse a supplied partition and its stored region statistics."""

    def __init__(self, regions: Sequence[Region]):
        self._regions = list(regions)

    def partition(
        self,
        oracle: CounterfactualOracle,
        rng: random.Random,
        deadline_cpu: Optional[float] = None,
    ) -> List[Region]:
        return list(self._regions)


def load_regions_from_csv(path) -> List[Region]:
    """Load regions and scoring statistics from a region-report CSV."""
    import csv

    regions: List[Region] = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            bounds = normalize_bounds(
                [tuple(int(v) for v in pair.split(":")) for pair in row["bounds"].split(";")]
            )
            history = row.get("split_history") or ""
            regions.append(
                Region(
                    region_id=int(row["region_id"]),
                    parent_id=int(row.get("parent_id") or -1),
                    bounds=bounds,
                    depth=int(row.get("depth") or 0),
                    size=region_size(bounds),
                    stats=RegionStats(
                        target_samples=int(row["target_samples"]),
                        samples=int(row["actual_samples"]),
                        violations=int(row["violations"]),
                    ),
                    split_history=history.split("|") if history else [],
                )
            )
    return regions
