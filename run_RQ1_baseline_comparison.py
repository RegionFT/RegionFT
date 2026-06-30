"""Run the RQ1 baseline-comparison experiment matrix."""

import os

# Each process runs one experiment, so numerical libraries use one thread.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse

from Experiments.common.datasets import PROTECTED_ATTRIBUTES
from Experiments.common.experiment_jobs import ExperimentJob, run_jobs
from Experiments.common.paths import RQ1_RESULTS_DIR

OUT_ROOT = RQ1_RESULTS_DIR

# RegionFT configuration evaluated in the baseline comparison.
REGIONFT_CONFIG = {
    "max_depth": 10,
    "min_samples_per_region": 1000,
    "max_samples_per_region": 3000,
    "min_gain": 0.0,
    "gini_weight_mode": "sample",
    "max_split_ratio": 1.0,
    "sampling_risk_focus_alpha": 1.0,
    "sampling_risk_floor": 0.00001,
    "batch_size": 512,
}

METHODS = ["aft", "expga", "grft", "limi", "sg", "regionft"]
MODELS = ["GBDT", "MLP", "LogReg", "DecTree"]
DEFAULT_RUNTIME = 1200
DEFAULT_REPEATS = 5
DEFAULT_PROCESSES = 64


def build_jobs(out_root, methods, models, datasets, runtime, repeat, cut_dir):
    jobs = []
    for method in methods:
        for dataset in datasets:
            for protected_pair in PROTECTED_ATTRIBUTES[dataset]:
                for model in models:
                    for repeat_id in range(repeat):
                        jobs.append(
                            ExperimentJob(
                                dataset=dataset,
                                model=model,
                                protected_pair=protected_pair,
                                method=method,
                                runtime=runtime,
                                repeat_id=repeat_id,
                                out_root=str(out_root),
                                config=REGIONFT_CONFIG,
                                cut_dir=cut_dir,
                            )
                        )
    return jobs


def run_matrix(
    out_root,
    methods,
    models,
    datasets,
    runtime,
    repeat,
    processes,
    cut_dir,
    no_analyze,
):
    """Run all selected cells and write their per-run analysis data."""
    from Experiments.common.result_extraction import write_long

    jobs = build_jobs(out_root, methods, models, datasets, runtime, repeat, cut_dir)
    print(f"{out_root.name}: {len(jobs)} runs -> {out_root}")
    print(f"  methods={methods} models={models} datasets={datasets}")
    print(f"  runtime={runtime}s repeat={repeat} processes={processes} cut_dir={cut_dir}")

    rows = run_jobs(jobs, processes, extract=not no_analyze)

    if no_analyze:
        print("  (--no-analyze: skipped result extraction)")
        return

    completed_rows = [row for row in rows if row]
    path = write_long(out_root, completed_rows, PROTECTED_ATTRIBUTES, MODELS, METHODS)
    print(f"  sensitivity_long -> {path}  ({len(completed_rows)} runs)")


def main():
    parser = argparse.ArgumentParser(description="Run the baseline-comparison experiment matrix.")
    parser.add_argument(
        "--processes",
        type=int,
        default=DEFAULT_PROCESSES,
        help="parallel worker processes",
    )
    parser.add_argument(
        "--runtime",
        type=int,
        default=DEFAULT_RUNTIME,
        help="per-run budget in seconds",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=DEFAULT_REPEATS,
        help="repeats per cell",
    )
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(PROTECTED_ATTRIBUTES),
        choices=list(PROTECTED_ATTRIBUTES),
    )
    parser.add_argument(
        "--no-analyze",
        action="store_true",
        help="run only; skip sensitivity_long.csv and cells/",
    )
    args = parser.parse_args()

    run_matrix(
        out_root=OUT_ROOT,
        methods=args.methods,
        models=args.models,
        datasets=args.datasets,
        runtime=args.runtime,
        repeat=args.repeat,
        processes=args.processes,
        cut_dir=None,
        no_analyze=args.no_analyze,
    )


if __name__ == "__main__":
    main()
