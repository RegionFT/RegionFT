"""Generate the final readable RQ results from per-run long CSV files."""

import argparse
from pathlib import Path

from Experiments.common.paths import (
    ARCHIVED_RESULTS_DIR,
    RQ1_RESULTS_DIR,
    RQ2_RQ3_RESULTS_DIR,
)
from Experiments.scripts.analyze_RQ_baseline_comparison import (
    analyze_long_results as analyze_baseline_results,
)
from Experiments.scripts.analyze_RQ_regionft import (
    analyze_long_results as analyze_regionft_results,
)


REGIONFT_VIEWS = {
    "partition": (
        "RQ2_partition_rule_long.csv",
        "RQ2_partition_rule_summary.csv",
    ),
    "depth": (
        "RQ2_max_depth_long.csv",
        "RQ2_max_depth_summary.csv",
    ),
    "alpha": (
        "RQ3_alpha_long.csv",
        "RQ3_alpha_summary.csv",
    ),
}


def _long_files(analysis_dir):
    return sorted(Path(analysis_dir).glob("sensitivity_long*.csv"))


def _require(paths, description):
    paths = list(paths)
    if not paths:
        raise FileNotFoundError(f"no {description} found")
    return paths


def analyze_retained_results(input_dir, output_dir):
    """Regenerate summaries from the long CSV files retained in ``Results``."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    written = analyze_baseline_results(
        [input_dir / "RQ1_baseline_long.csv"],
        output_dir / "RQ1_baseline_summary.csv",
        output_dir / "RQ1_significance.csv",
    )
    for view, (long_name, summary_name) in REGIONFT_VIEWS.items():
        written.extend(
            analyze_regionft_results(
                [input_dir / long_name],
                view,
                output_dir / summary_name,
            )
        )
    return written


def analyze_experiment_outputs(experiment_root, output_dir):
    """Finalize a newly completed experiment-results tree."""
    experiment_root = Path(experiment_root)
    output_dir = Path(output_dir)

    baseline_long = _require(
        _long_files(
            experiment_root
            / RQ1_RESULTS_DIR.name
            / "analysis"
        ),
        "baseline-comparison sensitivity_long CSVs",
    )
    regionft_long = _require(
        _long_files(experiment_root / RQ2_RQ3_RESULTS_DIR.name / "analysis"),
        "RegionFT-ablation sensitivity_long CSVs",
    )

    written = analyze_baseline_results(
        baseline_long,
        output_dir / "RQ1_baseline_summary.csv",
        output_dir / "RQ1_significance.csv",
        long_output=output_dir / "RQ1_baseline_long.csv",
    )
    for view, (long_name, summary_name) in REGIONFT_VIEWS.items():
        written.extend(
            analyze_regionft_results(
                regionft_long,
                view,
                output_dir / summary_name,
                long_output=output_dir / long_name,
            )
        )
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate the final RQ summary CSV files.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ARCHIVED_RESULTS_DIR,
        help="directory containing the retained RQ*_long.csv files",
    )
    parser.add_argument(
        "--from-experiments",
        type=Path,
        metavar="DIR",
        help="analyze a newly completed experiment-results tree",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARCHIVED_RESULTS_DIR,
        help="directory for the final long, summary, and significance CSV files",
    )
    args = parser.parse_args(argv)

    try:
        if args.from_experiments is None:
            written = analyze_retained_results(args.input_dir, args.output_dir)
        else:
            written = analyze_experiment_outputs(
                args.from_experiments,
                args.output_dir,
            )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print("Generated final results:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
