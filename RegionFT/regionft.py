from __future__ import annotations

import csv
import logging
import random
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from RegionFT.core import Region, bounds_to_text
from RegionFT.oracle import CounterfactualOracle
from RegionFT.partition import GiniPartitioner, Partitioner, RandomPartitioner
from RegionFT.sampler import RegionSampler, SamplingResult


class RegionFT:
    """Region-guided black-box fairness testing in two phases.

    The classifier provides inclusive integer ``data_range`` bounds and
    ``predict(rows)``; protected attributes are zero-based indices.
    """

    def __init__(
        self,
        black_box_model,
        protected_list: Sequence[int],
        max_depth: int = 10,
        min_samples_per_region: int = 30,
        max_samples_per_region: int = 100,
        min_gain: float = 0.0,
        gini_weight_mode: str = "size",
        max_split_ratio: float = 1.0,
        sampling_risk_focus_alpha: float = 1.0,
        sampling_risk_floor: float = 0.00001,
        batch_size: int = 512,
        sample_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
        show_logging: bool = False,
        partition_mode: str = "gini",
        partitioner: Optional[Partitioner] = None,
        disc_phase: str = "all",
    ):
        self.black_box_model = black_box_model
        self.protected_list = [int(item) for item in protected_list]

        self.max_depth = int(max_depth)
        self.min_samples_per_region = int(min_samples_per_region)
        self.max_samples_per_region = int(max_samples_per_region)
        self.min_gain = float(min_gain)
        self.gini_weight_mode = str(gini_weight_mode).lower()
        self.max_split_ratio = float(max_split_ratio)
        self.sampling_risk_focus_alpha = float(sampling_risk_focus_alpha)
        if self.sampling_risk_focus_alpha < 0:
            raise ValueError("sampling_risk_focus_alpha must be non-negative")
        self.sampling_risk_floor = float(sampling_risk_floor)
        if self.sampling_risk_floor < 0:
            raise ValueError("sampling_risk_floor must be non-negative")
        self.batch_size = max(1, int(batch_size))
        self.sample_draws = None if sample_draws is None else int(sample_draws)
        if self.sample_draws is not None and self.sample_draws < 0:
            raise ValueError("sample_draws must be non-negative")
        self.disc_phase = str(disc_phase).lower()
        if self.disc_phase not in ("all", "generation"):
            raise ValueError("disc_phase must be 'all' or 'generation'")
        self._random_seed = random_seed
        self.rng = random.Random(random_seed)
        self.partition_mode = str(partition_mode).lower()
        partition_kwargs = dict(
            max_depth=self.max_depth,
            min_samples_per_region=self.min_samples_per_region,
            max_samples_per_region=self.max_samples_per_region,
            min_gain=self.min_gain,
            gini_weight_mode=self.gini_weight_mode,
            max_split_ratio=self.max_split_ratio,
            batch_size=self.batch_size,
        )
        if partitioner is not None:
            self.partitioner = partitioner
        elif self.partition_mode == "random":
            self.partitioner = RandomPartitioner(**partition_kwargs)
        else:
            self.partitioner = GiniPartitioner(**partition_kwargs)
        self.gini_weight_mode = getattr(self.partitioner, "gini_weight_mode", self.gini_weight_mode)
        self.max_split_ratio = getattr(self.partitioner, "max_split_ratio", self.max_split_ratio)

        self.regions: List[Region] = []
        self.sampling_result = SamplingResult()
        self.test_data: List[List[int]] = []
        self.disc_data: List[List[int]] = []
        self.generation_disc_data: List[List[int]] = []
        self.partition_no_test = 0
        self.partition_no_disc = 0
        self.generation_no_test = 0
        self.generation_no_disc = 0
        self.no_test = 0
        self.no_disc = 0
        self.real_time_consumed = 0.0
        self.cpu_time_consumed = 0.0
        self.partition_cpu_time = 0.0
        self.partition_real_time = 0.0
        self.generation_cpu_time = 0.0
        self.generation_real_time = 0.0
        self.n_regions = 0

        if show_logging:
            logging.basicConfig(format="", level=logging.INFO)
        else:
            logging.basicConfig(level=logging.CRITICAL + 1)

    def test(
        self,
        runtime: Optional[int] = None,
        label: Tuple[str, int] = ("regionft", 0),
        disc_save_to: Path = Path("DiscData"),
        test_save_to: Path = Path("TestData"),
        region_save_to: Optional[Path] = None,
    ):
        start_real_time = time.time()
        start_cpu_time = time.process_time()
        deadline_cpu = None
        if runtime is not None:
            deadline_cpu = start_cpu_time + float(runtime)

        # Partitioning phase: construct the final partition.
        logging.info("[RegionFT] start | runtime=%s", runtime)
        oracle = CounterfactualOracle(self.black_box_model, self.protected_list)
        self.regions = self.partitioner.partition(oracle, self.rng, deadline_cpu)
        self.partition_cpu_time = time.process_time() - start_cpu_time
        self.partition_real_time = time.time() - start_real_time
        self.partition_no_test = oracle.phase_tests.get("partition", 0)
        self.partition_no_disc = oracle.phase_violations.get("partition", 0)
        logging.info(
            "[RegionFT] partition done: %d regions | tests=%d idi=%d | cpu=%.1fs",
            len(self.regions),
            self.partition_no_test,
            self.partition_no_disc,
            time.process_time() - start_cpu_time,
        )

        # Test generation phase: select regions by size and estimated IDI rate.
        sampler = RegionSampler(
            sampling_risk_focus_alpha=self.sampling_risk_focus_alpha,
            sampling_risk_floor=self.sampling_risk_floor,
            batch_size=self.batch_size,
        )
        sample_draws = self.sample_draws
        if sample_draws is None and runtime is None:
            sample_draws = self.max_samples_per_region * len(self.regions)

        # A separate stream keeps generation randomness stable across partition depths.
        sampler_rng = random.Random(None if self._random_seed is None
                                    else self._random_seed + 1_000_003)
        self.sampling_result = sampler.sample(
            regions=self.regions,
            oracle=oracle,
            rng=sampler_rng,
            draws=sample_draws,
            deadline_cpu=deadline_cpu,
        )
        self.generation_no_test = oracle.phase_tests.get("generation", 0)
        self.generation_no_disc = oracle.phase_violations.get("generation", 0)
        logging.info(
            "[RegionFT] generation done: tests=%d idi=%d | cpu=%.1fs",
            self.generation_no_test,
            self.generation_no_disc,
            time.process_time() - start_cpu_time,
        )

        # Top-level counts include both phases; phase-specific counts remain available.
        self.real_time_consumed = time.time() - start_real_time
        self.cpu_time_consumed = time.process_time() - start_cpu_time
        self.generation_cpu_time = self.cpu_time_consumed - self.partition_cpu_time
        self.generation_real_time = self.real_time_consumed - self.partition_real_time
        self.n_regions = len(self.regions)
        self.test_data = oracle.tested_anchors
        self.disc_data = oracle.violating_pairs
        self.generation_disc_data = oracle.violating_pairs_by_phase.get("generation", [])
        self.no_test = oracle.tests
        self.no_disc = oracle.violations
        logging.info(
            "[RegionFT] done: no_test=%d no_disc=%d ratio=%.4f | cpu=%.1fs real=%.1fs",
            self.no_test,
            self.no_disc,
            self.no_disc / float(self.no_test or 1),
            self.cpu_time_consumed,
            self.real_time_consumed,
        )

        if test_save_to is not None:
            self._write_rows(Path(test_save_to) / f"{label[0]}-{label[1]}.npy", self.test_data)
        if disc_save_to is not None:
            disc_rows = (self.generation_disc_data if self.disc_phase == "generation"
                         else self.disc_data)
            self._write_rows(Path(disc_save_to) / f"{label[0]}-{label[1]}.npy", disc_rows)
        if region_save_to is not None:
            self._write_region_report(Path(region_save_to) / f"{label[0]}-{label[1]}.csv")

    def _write_rows(self, path: Path, rows: Sequence[Sequence[int]]):
        """Save rows compactly when the target directory exists."""

        if not path.parent.is_dir():
            return
        arr = np.asarray(rows, dtype=np.int64)
        if arr.size:
            peak = max(abs(int(arr.min())), abs(int(arr.max())))
            for dtype in (np.int16, np.int32):
                if peak <= np.iinfo(dtype).max:
                    arr = arr.astype(dtype, copy=False)
                    break
        np.save(path, arr)

    def _write_region_report(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        sampler = RegionSampler(
            sampling_risk_focus_alpha=self.sampling_risk_focus_alpha,
            sampling_risk_floor=self.sampling_risk_floor,
            batch_size=self.batch_size,
        )
        sampler_weights, sampler_used_fallback = sampler.weights(self.regions)
        sampler_probs, _ = sampler.probabilities(self.regions)

        with path.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "region_id",
                    "parent_id",
                    "depth",
                    "size",
                    "target_samples",
                    "actual_samples",
                    "violations",
                    "estimated_ratio",
                    "gini",
                    "best_attr",
                    "best_cut",
                    "best_gain",
                    "gini_weight_mode",
                    "max_split_ratio",
                    "sampler_weight",
                    "sampler_probability",
                    "sampling_risk_focus_alpha",
                    "sampling_risk_floor",
                    "partition_sample_mode",
                    "sampler_used_fallback",
                    "sampler_draws",
                    "sampler_violations",
                    "partition_tests_total",
                    "partition_violations_total",
                    "sampler_tests_total",
                    "sampler_violations_total",
                    "bounds",
                    "split_history",
                    "batch_size",
                ]
            )
            for region, sampler_weight, sampler_prob in zip(
                self.regions,
                sampler_weights,
                sampler_probs,
            ):
                writer.writerow(
                    [
                        region.region_id,
                        region.parent_id,
                        region.depth,
                        region.size,
                        region.stats.target_samples,
                        region.stats.samples,
                        region.stats.violations,
                        region.stats.ratio,
                        region.stats.gini,
                        "" if region.best_attr is None else region.best_attr,
                        "" if region.best_cut is None else region.best_cut,
                        region.best_gain,
                        self.gini_weight_mode,
                        self.max_split_ratio,
                        sampler_weight,
                        sampler_prob,
                        self.sampling_risk_focus_alpha,
                        self.sampling_risk_floor,
                        "fresh_region",
                        int(sampler_used_fallback),
                        self.sampling_result.selected_region_counts.get(region.region_id, 0),
                        self.sampling_result.selected_region_violations.get(region.region_id, 0),
                        self.partition_no_test,
                        self.partition_no_disc,
                        self.generation_no_test,
                        self.generation_no_disc,
                        bounds_to_text(region.bounds),
                        "|".join(region.split_history),
                        self.batch_size,
                    ]
                )
