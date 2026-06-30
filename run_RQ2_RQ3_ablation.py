"""Run paired RegionFT partition, depth, and alpha ablations for RQ2/RQ3.
The depth-10, alpha-1 Gini run is shared across all three views."""

import os

# Each process runs one experiment, so numerical libraries use one thread.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
from pathlib import Path

from Experiments.common.datasets import PROTECTED_ATTRIBUTES
from Experiments.common.experiment_jobs import ExperimentJob, run_jobs
from Experiments.common.method_execution import is_regionft
from Experiments.common.paths import RQ2_RQ3_RESULTS_DIR


OUT_ROOT = RQ2_RQ3_RESULTS_DIR

BASE_REGIONFT_CONFIG = {
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

MODELS = ["GBDT", "MLP", "LogReg", "DecTree"]

SWEEP_DRAWS = 5_000_000
SWEEP_RUNTIME = 7200
SWEEP_CELLS = [
    f"{dataset}:{protected}:{model}"
    for dataset, protected_attributes in PROTECTED_ATTRIBUTES.items()
    for protected, _ in protected_attributes
    for model in MODELS
]

DEFAULT_DEPTHS = [2, 4, 6, 8, 10, 12]
DEFAULT_ALPHAS = [0, 0.1, 0.25, 0.5, 0.75, 1, 2, 4, 8, 10]


def find_protected_pair(dataset, name):
    for protected_name, index in PROTECTED_ATTRIBUTES[dataset]:
        if protected_name == name:
            return protected_name, index
    raise KeyError(f"{name!r} is not a protected attribute of {dataset}")


def resolve_cells(cell_specs):
    """Parse DATASET:PROTECTED:MODEL experiment cells."""
    return [
        (dataset, find_protected_pair(dataset, protected), model)
        for dataset, protected, model in (
            specification.split(":") for specification in cell_specs
        )
    ]


def calibration_cells(cells):
    return [
        (dataset, protected[0], model)
        for dataset, protected, model in cells
    ]


def build_jobs(
    out_root,
    methods,
    cells,
    runtime,
    repeat,
    regionft_config,
):
    jobs = []
    for method in methods:
        for dataset, protected, model in cells:
            for repeat_id in range(repeat):
                # Pair variants by repeat and preserve nested depth partitions.
                config = dict(regionft_config, random_seed=repeat_id)
                jobs.append(
                    ExperimentJob(
                        dataset=dataset,
                        model=model,
                        protected_pair=protected,
                        method=method,
                        runtime=runtime,
                        repeat_id=repeat_id,
                        out_root=str(out_root),
                        config=config,
                    )
                )
    return jobs


def _alpha_token(alpha):
    if float(alpha).is_integer():
        return f"a{int(alpha)}"
    return "a" + str(alpha).replace(".", "p")


def ablation_methods(depths, alphas, include_random=True):
    """Return all ablation variants and the partition-calibration subset."""
    depth_methods = [f"regionft_d{int(depth)}" for depth in depths]
    alpha_methods = [
        f"regionft_{_alpha_token(alpha)}"
        for alpha in alphas
        if float(alpha) != 1.0
    ]
    random_methods = ["regionft_random"] if include_random else []

    methods = []
    seen = set()
    for method in depth_methods + alpha_methods + random_methods:
        if method not in seen:
            seen.add(method)
            methods.append(method)
    return methods, depth_methods + random_methods


def run_ablation_matrix(
    out_root,
    methods,
    cells,
    runtime,
    draws,
    repeat,
    processes,
    include_calibration,
    calibration_samples=5000,
    calibration_bins=10,
    no_analyze=False,
    calibration_methods=None,
):
    """Run the ablation matrix and write its per-run analysis artifacts."""
    for method in methods:
        if not is_regionft(method):
            raise SystemExit(f"not a RegionFT-family method: {method!r}")

    from Experiments.common.result_extraction import write_long

    base_config = dict(BASE_REGIONFT_CONFIG, disc_phase="generation")
    if draws is not None:
        base_config["sample_draws"] = draws

    jobs = build_jobs(
        out_root,
        methods,
        cells,
        runtime,
        repeat,
        base_config,
    )
    budget = f"draws={draws}" if draws else f"runtime={runtime}s"
    print(f"{Path(out_root).name}: {len(jobs)} runs -> {out_root}")
    print(
        f"  methods={methods}  cells={len(cells)}  {budget}  "
        f"repeat={repeat}  processes={processes}"
    )

    rows = run_jobs(jobs, processes, extract=not no_analyze)

    if no_analyze:
        print("  (--no-analyze: skipped result extraction and calibration)")
        return

    completed_rows = [row for row in rows if row]
    path = write_long(
        out_root,
        completed_rows,
        PROTECTED_ATTRIBUTES,
        MODELS,
        [],
    )
    print(f"  sensitivity_long -> {path}  ({len(completed_rows)} runs)")

    if include_calibration:
        from Experiments.scripts.calibrate_partitions import calibrate

        calibrate(
            out_root,
            calibration_methods or methods,
            calibration_cells(cells),
            n_samples=calibration_samples,
            bins=calibration_bins,
            repeats=1,
            processes=processes,
        )


def main():
    p = argparse.ArgumentParser(
        description="Run the RegionFT partition, depth, and alpha ablations."
    )
    p.add_argument("--processes", type=int, default=64)
    p.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=DEFAULT_DEPTHS,
        help="partition depths; depth 10 is always included",
    )
    p.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=DEFAULT_ALPHAS,
        help="risk-focus values evaluated on the depth-10 partition",
    )
    p.add_argument(
        "--cells",
        nargs="+",
        metavar="DATASET:PROTECTED:MODEL",
        default=SWEEP_CELLS,
        help="experiment cells",
    )
    p.add_argument(
        "--draws",
        type=int,
        default=SWEEP_DRAWS,
        help="generation budget; use 0 to rely on --runtime",
    )
    p.add_argument(
        "--runtime",
        type=int,
        default=SWEEP_RUNTIME,
        help="time budget or ceiling in seconds",
    )
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--calib-samples", type=int, default=5000)
    p.add_argument(
        "--no-analyze",
        action="store_true",
        help="run only; skip result extraction and calibration",
    )
    args = p.parse_args()

    depths = sorted(set(args.depths) | {10})
    methods, calibration_methods = ablation_methods(depths, args.alphas)
    cells = resolve_cells(args.cells)
    run_ablation_matrix(
        out_root=OUT_ROOT,
        methods=methods,
        cells=cells,
        runtime=args.runtime,
        draws=args.draws or None,
        repeat=args.repeat,
        processes=args.processes,
        include_calibration=True,
        calibration_samples=args.calib_samples,
        calibration_methods=calibration_methods,
        no_analyze=args.no_analyze,
    )


if __name__ == "__main__":
    main()
