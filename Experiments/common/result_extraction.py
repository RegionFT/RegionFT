"""Extract compact analysis artifacts from completed experiment runs."""

import csv
import glob
from functools import cache
from pathlib import Path

import numpy as np

from Experiments.common.datasets import load_dataset
from Experiments.common.metrics import (
    cell_codes,
    count_idi,
    diversity,
    read_idi_rows,
    read_log,
)


COVERAGE_GRANULARITIES = (1, 2, 3, 4, 5)
DIVERSITY_SAMPLE_SIZES = (50, 100, 200, 500, 1000, 2000, 3000, 5000)
CONTAINMENT_GRANULARITIES = (2, 3, 4, 5)
DIVERSITY_REPEATS = 5

SENSITIVITY_COLUMNS = (
    "run_id", "dataset", "model", "protected", "method", "repeat",
    "n_idi", "n_test",
    "cov1", "cov2", "cov3", "cov4", "cov5",
    "div50", "div100", "div200", "div500",
    "div1000", "div2000", "div3000", "div5000",
    "real_time", "cpu_time", "partition_real", "partition_cpu",
    "generation_real", "generation_cpu", "n_regions",
)


@cache
def _data_ranges(dataset):
    ranges, _ = load_dataset(dataset)
    return ranges


def _disc_path(job):
    protected = job.protected_pair[0]
    pattern = (
        Path(job.out_root)
        / job.method
        / "disc"
        / f"{job.method}-{job.model}-{job.dataset}-{protected}-*-{job.repeat_id}"
    )
    matches = glob.glob(str(pattern) + ".npy") + glob.glob(str(pattern) + ".csv")
    return Path(sorted(matches)[0]) if matches else None


def _rounded(value, digits=2):
    return round(value, digits) if value is not None else ""


def _test_count(log, generation):
    config = log.get("config") or {}
    if (
        str(config.get("disc_phase", "all")).lower() == "generation"
        and generation.get("no_test") is not None
    ):
        return generation["no_test"]
    return log.get("no_test")


def _timing_values(log, generation):
    values = {
        "real_time": _rounded(log.get("real_time")),
        "cpu_time": _rounded(log.get("cpu_time")),
    }
    partition = log.get("partition") or {}
    if not partition and not generation:
        return values | {
            "partition_real": "",
            "partition_cpu": "",
            "generation_real": "",
            "generation_cpu": "",
            "n_regions": "",
        }

    n_regions = partition.get("n_regions")
    return values | {
        "partition_real": _rounded(partition.get("real_time")),
        "partition_cpu": _rounded(partition.get("cpu_time")),
        "generation_real": _rounded(generation.get("real_time")),
        "generation_cpu": _rounded(generation.get("cpu_time")),
        "n_regions": n_regions if n_regions is not None else "",
    }


def _coverage_values(idi_rows, ranges, protected_index):
    values = {}
    saved_codes = {}
    for granularity in COVERAGE_GRANULARITIES:
        codes = cell_codes(
            idi_rows,
            ranges,
            protected_index,
            granularity,
        )
        values[f"cov{granularity}"] = int(codes.shape[0])
        if granularity in CONTAINMENT_GRANULARITIES:
            saved_codes[granularity] = codes
    return values, saved_codes


def _diversity_values(idi_rows):
    return {
        f"div{sample_size}": _rounded(
            diversity(
                idi_rows,
                sample_size=sample_size,
                repeats=DIVERSITY_REPEATS,
            ),
            4,
        )
        for sample_size in DIVERSITY_SAMPLE_SIZES
    }


def _extract_row(job):
    disc_path = _disc_path(job)
    if disc_path is None or not disc_path.is_file():
        return None, {}

    idi_rows = read_idi_rows(disc_path)
    run_id = disc_path.stem
    log_path = Path(job.out_root) / job.method / "log" / f"{run_id}.json"
    log = read_log(log_path) if log_path.is_file() else {}

    generation = log.get("generation") or log.get("sampling") or {}
    row = {
        "run_id": run_id,
        "dataset": job.dataset,
        "model": job.model,
        "protected": job.protected_pair[0],
        "method": job.method,
        "repeat": str(job.repeat_id),
        "n_idi": count_idi(idi_rows),
        "n_test": _test_count(log, generation),
    }
    row.update(_timing_values(log, generation))
    ranges = _data_ranges(job.dataset)
    coverage, saved_codes = _coverage_values(
        idi_rows,
        ranges,
        job.protected_pair[1],
    )
    row.update(coverage)
    row.update(_diversity_values(idi_rows))
    return row, saved_codes


def _save_cells(out_root, run_id, codes):
    cells_dir = Path(out_root) / "analysis" / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"g{granularity}": np.asarray(values, dtype=np.int64)
        for granularity, values in codes.items()
    }
    np.savez_compressed(cells_dir / f"{run_id}.npz", **arrays)


def extract_after_run(job):
    """Collect one completed run into a long row and a cells archive."""
    row, codes = _extract_row(job)
    if row is not None and codes:
        _save_cells(job.out_root, row["run_id"], codes)
    return row


def _row_order(protected_attributes, models, methods):
    dataset_rank = {
        name: index
        for index, name in enumerate(protected_attributes)
    }
    model_rank = {name: index for index, name in enumerate(models)}
    method_rank = {name: index for index, name in enumerate(methods)}

    def row_key(row):
        protected = [
            name
            for name, _ in protected_attributes.get(row["dataset"], ())
        ]
        protected_rank = (
            protected.index(row["protected"])
            if row["protected"] in protected
            else len(protected)
        )
        try:
            repeat = int(row.get("repeat", 0))
        except (TypeError, ValueError):
            repeat = 0
        return (
            dataset_rank.get(row["dataset"], 99),
            model_rank.get(row["model"], 99),
            protected_rank,
            method_rank.get(row["method"], 99),
            row["method"],
            repeat,
        )

    return row_key


def write_long(out_root, rows, protected_attributes, models, methods):
    """Write per-run rows in the experiment's canonical order."""
    analysis_dir = Path(out_root) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    path = analysis_dir / "sensitivity_long.csv"
    with path.open("w", newline="") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=SENSITIVITY_COLUMNS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(
            sorted(
                (row for row in rows if row),
                key=_row_order(protected_attributes, models, methods),
            )
        )
    return path
