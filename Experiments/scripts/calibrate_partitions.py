"""Estimate saved regions' IDI rates with fresh uniform samples."""
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Calibration parallelizes by process.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import glob
import traceback
import zlib
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from RegionFT.oracle import CounterfactualOracle
from RegionFT.partition import load_regions_from_csv
from Experiments.common.datasets import PROTECTED_ATTRIBUTES
from Experiments.common.model import load_cut
from Experiments.common.paths import RQ2_RQ3_RESULTS_DIR


MODELS = ["GBDT", "MLP", "LogReg", "DecTree"]
DATASET_PROTECTED = PROTECTED_ATTRIBUTES
DEFAULT_N_SAMPLES = 5000
DEFAULT_BINS = 10
# Alpha variants share the depth-10 partition, so it is calibrated only once.
DEFAULT_CALIB_METHODS = [
    f"regionft_d{depth}" for depth in (2, 4, 6, 8, 10, 12)
] + ["regionft_random"]

REGION_COLUMNS = (
    "region_id", "size", "estimated_ratio", "calibrated_ratio", "calib_draws",
)
HISTOGRAM_COLUMNS = (
    "method", "dataset", "model", "protected", "n_samples", "repeats",
    "n_regions", "mean_ratio", "wmean_ratio",
)


def _protected_pair(dataset, name):
    for protected_name, index in DATASET_PROTECTED[dataset]:
        if protected_name == name:
            return protected_name, index
    raise KeyError(f"{name!r} is not a protected attribute of {dataset}")


def resolve_cells(cells_spec, datasets, models):
    if cells_spec:
        cells = []
        for specification in cells_spec:
            dataset, protected, model = specification.split(":")
            _protected_pair(dataset, protected)
            cells.append((dataset, protected, model))
        return cells
    return [
        (dataset, protected, model)
        for dataset in datasets
        for protected, _ in DATASET_PROTECTED[dataset]
        for model in models
    ]


def _cell_stems(method_dir, method, model, dataset, protected):
    pat = method_dir / "log" / f"{method}-{model}-{dataset}-{protected}-*-*.json"
    return sorted(Path(p).stem for p in glob.glob(str(pat)))


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sample_region(region, n_samples, rng):
    low = np.array([int(lo) for lo, _ in region.bounds], dtype=np.int64)
    span = np.array(
        [int(hi) - int(lo) + 1 for lo, hi in region.bounds],
        dtype=np.int64,
    )
    return low + (rng.random((n_samples, low.size)) * span).astype(np.int64)


def calibrate_one(task):
    """Calibrate one saved partition and return data for its histogram."""
    out_root, method, dataset, model, protected, stem, n_samples = task
    try:
        method_dir = Path(out_root) / method
        report = method_dir / "partition" / f"{stem}.csv"
        if not report.is_file():
            return None
        regions = load_regions_from_csv(str(report))
        if not regions:
            return None

        protected_idx = _protected_pair(dataset, protected)[1]
        cut, _ = load_cut(dataset, model)
        oracle = CounterfactualOracle(cut, [protected_idx])
        # A stable seed makes recalibrating the same partition reproducible.
        rng = np.random.default_rng(zlib.crc32(stem.encode()) & 0xFFFFFFFF)

        rows, ratios, sizes = [], [], []
        for region in regions:
            anchors = _sample_region(region, n_samples, rng)
            checks = oracle.check_many(anchors.tolist(), record=False, phase="calib")
            violations = np.fromiter(
                (check.is_violation for check in checks),
                dtype=bool,
                count=len(checks),
            )
            ratio = float(violations.mean())
            ratios.append(ratio)
            sizes.append(int(region.size))
            rows.append(
                {
                    "region_id": region.region_id,
                    "size": region.size,
                    "estimated_ratio": region.stats.ratio,
                    "calibrated_ratio": ratio,
                    "calib_draws": n_samples,
                }
            )

        output = (Path(out_root) / "analysis" / "calibration" / method
                  / f"{stem}.csv")
        _write_csv(output, REGION_COLUMNS, rows)
        return (method, dataset, model, protected), ratios, sizes
    except Exception:
        sys.stderr.write(f"[calibrate] FAILED {task[:6]}\n")
        traceback.print_exc()
        return None


def build_tasks(out_root, methods, cells, n_samples, repeats):
    """Create one task for each selected saved partition."""
    tasks = []
    for method in methods:
        method_dir = Path(out_root) / method
        for dataset, protected, model in cells:
            stems = _cell_stems(method_dir, method, model, dataset, protected)
            tasks.extend(
                (str(out_root), method, dataset, model, protected, stem, n_samples)
                for stem in stems[:repeats]
            )
    return tasks


def _histogram_rows(results, n_samples, bins):
    """Build region-count and region-size-weighted histograms for each cell."""
    pooled_ratios = defaultdict(list)
    pooled_sizes = defaultdict(list)
    repeats_seen = defaultdict(int)
    for result in results:
        if result is None:
            continue
        key, ratios, sizes = result
        pooled_ratios[key].extend(ratios)
        pooled_sizes[key].extend(sizes)
        repeats_seen[key] += 1

    edges = np.linspace(0.0, 1.0, bins + 1)
    count_columns = [f"n_b{i}" for i in range(bins)]
    weight_columns = [f"w_b{i}" for i in range(bins)]
    rows = []
    for key, ratios in pooled_ratios.items():
        method, dataset, model, protected = key
        ratios = np.asarray(ratios, dtype=float)
        sizes = np.asarray(pooled_sizes[key], dtype=float)
        total_size = sizes.sum()

        count_histogram, _ = np.histogram(ratios, bins=edges)
        weighted_histogram, _ = np.histogram(ratios, bins=edges, weights=sizes)
        weighted_fractions = (
            weighted_histogram / total_size
            if total_size > 0
            else np.zeros_like(weighted_histogram)
        )
        weighted_mean = (
            round(float((ratios * sizes).sum() / total_size), 4)
            if total_size > 0
            else 0.0
        )

        row = {
            "method": method,
            "dataset": dataset,
            "model": model,
            "protected": protected,
            "n_samples": n_samples,
            "repeats": repeats_seen[key],
            "n_regions": int(ratios.size),
            "mean_ratio": round(float(ratios.mean()), 4),
            "wmean_ratio": weighted_mean,
        }
        for column, count in zip(count_columns, count_histogram.tolist()):
            row[column] = count
        for column, fraction in zip(weight_columns, weighted_fractions.tolist()):
            row[column] = round(fraction, 6)
        rows.append(row)
    rows.sort(key=lambda row: (
        row["method"], row["dataset"], row["protected"], row["model"],
    ))
    return rows, count_columns, weight_columns


def calibrate(
    out_root,
    methods,
    cells,
    n_samples=DEFAULT_N_SAMPLES,
    bins=DEFAULT_BINS,
    repeats=1,
    processes=12,
):
    """Calibrate ``(dataset, protected, model)`` cells and write their CSVs."""
    out_root = Path(out_root)
    calibration_dir = out_root / "analysis" / "calibration"
    tasks = build_tasks(out_root, methods, cells, n_samples, repeats)
    print(f"calibration: {len(tasks)} partitions (N={n_samples}) -> {calibration_dir}")
    if not tasks:
        return []

    calibration_dir.mkdir(parents=True, exist_ok=True)
    worker_count = max(1, min(processes, len(tasks)))
    with Pool(processes=worker_count) as pool:
        results = pool.map(calibrate_one, tasks)

    rows, count_columns, weight_columns = _histogram_rows(results, n_samples, bins)
    histogram_path = calibration_dir / "histogram.csv"
    _write_csv(
        histogram_path,
        HISTOGRAM_COLUMNS + tuple(count_columns) + tuple(weight_columns),
        rows,
    )

    print(
        f"histogram ({bins} bins over [0,1]; "
        "n_b=region count, w_b=size-weighted fraction) "
        f"-> {histogram_path}  ({len(rows)} cells)"
    )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Uniformly calibrate saved RegionFT partitions.")
    parser.add_argument("--out-root",
                        default=str(RQ2_RQ3_RESULTS_DIR),
                        help="results tree whose saved partitions to calibrate")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_CALIB_METHODS,
                        help="methods to calibrate (default: depth variants and random)")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_PROTECTED),
                        choices=list(DATASET_PROTECTED))
    parser.add_argument("--cells", nargs="+", metavar="DATASET:PROTECTED:MODEL",
                        help="explicit cells (overrides --datasets/--models)")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES,
                        help="uniform samples per region")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS,
                        help="equal-width histogram bins over [0,1]")
    parser.add_argument("--repeats", type=int, default=1,
                        help="saved partitions calibrated per cell")
    parser.add_argument("--processes", type=int, default=12,
                        help="parallel worker processes")
    args = parser.parse_args()

    cells = resolve_cells(args.cells, args.datasets, args.models)
    calibrate(args.out_root, args.methods, cells, args.n_samples, args.bins,
              args.repeats, args.processes)


if __name__ == "__main__":
    main()
