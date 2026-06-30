"""Run one fairness-testing configuration from the CLI or experiment scripts."""
import os

# Each experiment job receives one CPU core; set this before importing NumPy.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np

from Experiments.common.datasets import DATASET_FILES
from Experiments.common.method_execution import (
    MethodRun,
    is_regionft,
    regionft_variant_overrides,
    run_method,
)
from Experiments.common.model import load_cut
from Experiments.common.paths import (
    ARCHIVED_RESULTS_DIR,
    CUT_DIR,
    DATASETS_DIR,
)


def _count_pairs(path):
    """Count two-row instance pairs in a saved disc or test file."""
    path = Path(path)
    if not path.is_file():
        return None
    if path.suffix == ".npy":
        return int(np.load(path, mmap_mode="r").shape[0]) // 2
    with path.open("rb") as fh:
        lines = sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(65536), b""))
    return lines // 2


def _build_record(tester, method, model_name, dataset_name, protected_name,
                  repeat, runtime, label, disc_dir, test_dir, regionft_config):
    """Build the JSON record saved for one repeat."""
    no_disc = getattr(tester, "no_disc", None)
    if no_disc is None:
        no_disc = _count_pairs(Path(disc_dir) / f"{label[0]}-{label[1]}.npy") or 0
    no_test = getattr(tester, "no_test", None)
    if no_test is None and test_dir is not None:
        no_test = _count_pairs(Path(test_dir) / f"{label[0]}-{label[1]}.npy")
    no_test = no_test or 0

    record = {
        "label": label[0],
        "repeat": repeat,
        "method": method,
        "model": model_name,
        "dataset": dataset_name,
        "protected": protected_name,
        "runtime": runtime,
        "no_disc": no_disc,
        "no_test": no_test,
        "prec": (no_disc / no_test) if no_test else 0.0,
        "cpu_time": getattr(tester, "cpu_time_consumed", None),
        "real_time": getattr(tester, "real_time_consumed", None),
    }

    if hasattr(tester, "local_cpu_time_consumed"):
        record["local_cpu_time"] = tester.local_cpu_time_consumed
        record["local_real_time"] = tester.local_real_time_consumed

    if is_regionft(method):
        record["partition"] = {
            "cpu_time": getattr(tester, "partition_cpu_time", None),
            "real_time": getattr(tester, "partition_real_time", None),
            "no_test": getattr(tester, "partition_no_test", None),
            "no_disc": getattr(tester, "partition_no_disc", None),
            "n_regions": getattr(tester, "n_regions", None),
        }
        record["generation"] = {
            "cpu_time": getattr(tester, "generation_cpu_time", None),
            "real_time": getattr(tester, "generation_real_time", None),
            "no_test": getattr(tester, "generation_no_test", None),
            "no_disc": getattr(tester, "generation_no_disc", None),
        }
        record["config"] = regionft_config
    return record


def _output_dirs(out_root, method, save_test):
    method_dir = Path(out_root) / method
    disc_dir = method_dir / "disc"
    log_dir = method_dir / "log"
    disc_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    partition_dir = None
    if is_regionft(method):
        partition_dir = method_dir / "partition"
        partition_dir.mkdir(parents=True, exist_ok=True)

    test_dir = None
    if save_test or method == "limi":
        test_dir = method_dir / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
    return disc_dir, log_dir, partition_dir, test_dir


def _budget_label(method, runtime, regionft_config):
    sample_draws = (
        (regionft_config or {}).get("sample_draws")
        if is_regionft(method)
        else None
    )
    return f"d{sample_draws}" if sample_draws is not None else str(runtime)


def _write_json(path, record):
    with path.open("w") as file_handle:
        json.dump(record, file_handle, indent=2)


def run_cell(
    dataset_name,
    model_name,
    protected_pair,
    method,
    runtime,
    repeat,
    out_root,
    start_label=0,
    show_logging=False,
    save_test=False,
    regionft_config=None,
    cut_dir=None,
):
    """Run repeated trials for one dataset, model, attribute, and method."""
    protected_name, protected_index = protected_pair
    if is_regionft(method):
        regionft_config = {
            **(regionft_config or {}),
            **regionft_variant_overrides(method),
        }

    disc_dir, log_dir, partition_dir, test_dir = _output_dirs(
        out_root,
        method,
        save_test,
    )
    cut, dataframe = load_cut(dataset_name, model_name, cut_dir)
    original_data = dataframe.iloc[:, :-1].values.astype(float)
    budget = _budget_label(method, runtime, regionft_config)

    for repeat_id in range(start_label, start_label + repeat):
        label = (f"{method}-{model_name}-{dataset_name}-{protected_name}-{budget}", repeat_id)
        try:
            tester = run_method(
                MethodRun(
                    method=method,
                    cut=cut,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    protected_name=protected_name,
                    protected_index=protected_index,
                    original_data=original_data,
                    runtime=runtime,
                    label=label,
                    disc_dir=disc_dir,
                    test_dir=test_dir,
                    partition_dir=partition_dir,
                    show_logging=show_logging,
                    regionft_config=regionft_config,
                    cut_dir=cut_dir,
                )
            )
            if tester is None:
                continue
            record = _build_record(
                tester,
                method,
                model_name,
                dataset_name,
                protected_name,
                repeat_id,
                runtime,
                label,
                disc_dir,
                test_dir,
                regionft_config,
            )
            _write_json(log_dir / f"{label[0]}-{label[1]}.json", record)
        except Exception as exc:
            error = {
                "label": label[0], "repeat": repeat_id, "method": method,
                "model": model_name, "dataset": dataset_name, "protected": protected_name,
                "runtime": runtime, "error": repr(exc), "traceback": traceback.format_exc(),
            }
            try:
                _write_json(
                    log_dir / f"{label[0]}-{label[1]}.error.json",
                    error,
                )
            except Exception:
                pass
            print(f"[run_cell] FAILED {label[0]}-{repeat_id}: {exc!r}", file=sys.stderr)


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _non_negative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def _finite_float(value):
    value = float(value)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be a finite number")
    return value


def _non_negative_float(value):
    value = _finite_float(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return value


def _unit_float(value):
    value = _finite_float(value)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("must be a finite number between 0 and 1")
    return value


# Standalone CLI defaults, matching the baseline comparison.
_REGIONFT_CLI_DEFAULTS = {
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


def _build_parser():
    parser = argparse.ArgumentParser(description="Run one fairness-testing configuration.")
    parser.add_argument(
        "--dataset", required=True, choices=("Adult", "Credit", "Bank", "Lsac"),
    )
    parser.add_argument("--protected", required=True)
    parser.add_argument(
        "--model", required=True, choices=("GBDT", "MLP", "LogReg", "DecTree"),
    )
    parser.add_argument(
        "--method", required=True,
        choices=("aft", "vbtx", "vbt", "themis", "expga", "sg", "regionft", "limi", "grft"),
    )
    parser.add_argument(
        "--runtime", type=_positive_int, default=60,
        help="time budget in seconds (default: 60)",
    )
    parser.add_argument(
        "--repeat", type=_positive_int, default=1,
        help="number of runs (default: 1)",
    )
    parser.add_argument(
        "--output", type=Path,
        help="output directory (default: Results/runner_examples/run_<number>)",
    )
    parser.add_argument(
        "--show-log",
        action="store_true",
        help="show progress in the terminal",
    )

    defaults = _REGIONFT_CLI_DEFAULTS
    regionft = parser.add_argument_group(
        "RegionFT options",
        "Override the evaluation configuration; ignored for other methods.",
    )
    regionft.add_argument(
        "--max-depth", type=_non_negative_int,
        help=f"maximum partition depth (default: {defaults['max_depth']})",
    )
    regionft.add_argument(
        "--min-samples-per-region", type=_non_negative_int,
        help=f"minimum partition samples per region (default: {defaults['min_samples_per_region']})",
    )
    regionft.add_argument(
        "--max-samples-per-region", type=_non_negative_int,
        help=f"maximum partition samples per region (default: {defaults['max_samples_per_region']})",
    )
    regionft.add_argument(
        "--min-gain", type=_finite_float,
        help=f"minimum Gini gain for splitting (default: {defaults['min_gain']})",
    )
    regionft.add_argument(
        "--gini-weight-mode", choices=("size", "sample"),
        help=f"child weighting for Gini gain (default: {defaults['gini_weight_mode']})",
    )
    regionft.add_argument(
        "--max-split-ratio", type=_unit_float,
        help=f"do not split regions at or above this IDI ratio (default: {defaults['max_split_ratio']})",
    )
    regionft.add_argument(
        "--sampling-risk-focus-alpha", type=_non_negative_float,
        help=f"risk-focus exponent (default: {defaults['sampling_risk_focus_alpha']})",
    )
    regionft.add_argument(
        "--sampling-risk-floor", type=_non_negative_float,
        help=f"risk floor used in sampling weights (default: {defaults['sampling_risk_floor']})",
    )
    regionft.add_argument(
        "--batch-size", type=_positive_int,
        help=f"CheckIDI batch size (default: {defaults['batch_size']})",
    )
    return parser


def _regionft_config_from_args(args, parser):
    overrides = {
        name: getattr(args, name)
        for name in _REGIONFT_CLI_DEFAULTS
        if getattr(args, name) is not None
    }

    if args.method != "regionft":
        if overrides:
            print(
                f"Ignoring RegionFT options because --method is {args.method}.",
                file=sys.stderr,
            )
        return None

    config = _REGIONFT_CLI_DEFAULTS | overrides
    if config["max_samples_per_region"] < config["min_samples_per_region"]:
        parser.error("--max-samples-per-region must be >= --min-samples-per-region")
    return config


def _feature_index(dataset_path, feature_name, parser):
    with dataset_path.open(newline="") as fh:
        feature_names = next(csv.reader(fh))[:-1]

    if feature_name not in feature_names:
        parser.error(f"{feature_name!r} is not a feature of {dataset_path.stem}")

    return feature_names.index(feature_name)


def _next_example_dir():
    output_root = ARCHIVED_RESULTS_DIR / "runner_examples"
    run_number = 1
    while (output_root / f"run_{run_number:03d}").exists():
        run_number += 1
    return output_root / f"run_{run_number:03d}"


def _run_ids(args):
    stem = (
        f"{args.method}-{args.model}-{args.dataset}-"
        f"{args.protected}-{args.runtime}"
    )
    return [f"{stem}-{repeat_id}" for repeat_id in range(args.repeat)]


def _existing_run_output(method_dir, run_ids):
    return next(
        (
            path
            for run_id in run_ids
            for path in method_dir.glob(f"*/{run_id}.*")
        ),
        None,
    )


def _check_run_outputs(method_dir, run_ids):
    for run_id in run_ids:
        log_path = method_dir / "log" / f"{run_id}.json"
        error_path = method_dir / "log" / f"{run_id}.error.json"
        if error_path.is_file():
            print(f"Run failed; see {error_path}", file=sys.stderr)
            return 1
        if not log_path.is_file():
            print(f"Run did not produce {log_path}", file=sys.stderr)
            return 1
    return 0


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    regionft_config = _regionft_config_from_args(args, parser)

    dataset_path = Path(DATASETS_DIR) / DATASET_FILES[args.dataset]
    model_path = Path(CUT_DIR) / f"{args.model}{args.dataset}.joblib"
    if not dataset_path.is_file():
        parser.error(f"dataset not found: {dataset_path}")
    if not model_path.is_file():
        parser.error(f"model not found: {model_path}")
    protected_index = _feature_index(dataset_path, args.protected, parser)

    output = args.output.resolve() if args.output else _next_example_dir()
    if output.exists() and not output.is_dir():
        parser.error(f"output is not a directory: {output}")

    main_experiments = (ARCHIVED_RESULTS_DIR / "main_experiments").resolve()
    if output.is_relative_to(main_experiments):
        parser.error("--output cannot be inside Results/main_experiments")

    run_ids = _run_ids(args)
    method_dir = output / args.method
    existing = _existing_run_output(method_dir, run_ids)
    if existing:
        parser.error(f"run output already exists: {existing}")

    print(
        f"Running {args.method}: dataset={args.dataset}, protected={args.protected}, "
        f"model={args.model}, runtime={args.runtime}s, repeat={args.repeat}\n"
        f"Output: {output}",
        flush=True,
    )
    run_cell(
        dataset_name=args.dataset,
        model_name=args.model,
        protected_pair=(args.protected, protected_index),
        method=args.method,
        runtime=args.runtime,
        repeat=args.repeat,
        out_root=output,
        show_logging=args.show_log,
        regionft_config=regionft_config,
    )

    if _check_run_outputs(method_dir, run_ids):
        return 1

    print(f"Completed {len(run_ids)} run(s). Results saved to {method_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
