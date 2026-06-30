"""Summarize the RegionFT depth, alpha, and partition ablations."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import re
import statistics
from collections import defaultdict

from Experiments.common.paths import RQ2_RQ3_RESULTS_DIR


VALUE_COLUMNS = (
    "n_idi", "n_test", "cov2", "cov3", "cov4", "cov5", "n_regions",
    "partition_cpu", "generation_cpu", "real_time", "cpu_time",
)
VIEW_FILENAMES = {
    "depth": "depth_summary.csv",
    "alpha": "alpha_summary.csv",
    "partition": "partition_summary.csv",
}

_DEPTH_PATTERN = re.compile(r"^regionft_d(\d+)$")
_ALPHA_PATTERN = re.compile(r"^regionft_a(.+)$")


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values):
    values = [value for value in values if value is not None]
    return round(statistics.fmean(values), 4) if values else ""


def read_sensitivity_rows(paths):
    """Read compatible per-run CSV files into one table."""
    rows = []
    header = None
    for path in map(Path, paths):
        with path.open(newline="") as file_handle:
            reader = csv.DictReader(file_handle)
            if reader.fieldnames:
                if header is not None and reader.fieldnames != header:
                    raise ValueError(f"incompatible long CSV columns: {path}")
                header = reader.fieldnames
            rows.extend(reader)
    return rows, header


def _cell_means(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["model"], row["protected"], row["method"])
        groups[key].append(row)

    means = {}
    for key, members in groups.items():
        means[key] = {
            column: _mean([_number(row.get(column)) for row in members])
            for column in VALUE_COLUMNS
        }
        means[key]["repeats"] = len(members)
    return means


def _alpha_value(token):
    return float(token.replace("p", "."))


def _group_views(rows):
    views = {name: defaultdict(dict) for name in VIEW_FILENAMES}

    for (dataset, model, protected, method), values in _cell_means(rows).items():
        cell = (dataset, model, protected)
        depth_match = _DEPTH_PATTERN.match(method)
        alpha_match = _ALPHA_PATTERN.match(method)

        if depth_match:
            depth = int(depth_match.group(1))
            views["depth"][cell][depth] = values
            if depth == 10:
                # The depth-10 run is also alpha=1 and the Gini partition.
                views["alpha"][cell][1.0] = values
                views["partition"][cell]["gini"] = values
        elif alpha_match:
            views["alpha"][cell][_alpha_value(alpha_match.group(1))] = values
        elif method == "regionft_random":
            views["partition"][cell]["random"] = values

    return views


def select_view_rows(rows, view):
    """Select the long rows used by one ablation view."""
    if view == "depth":
        return [row for row in rows if _DEPTH_PATTERN.match(row["method"])]
    if view == "alpha":
        return [row for row in rows
                if (_ALPHA_PATTERN.match(row["method"])
                    or row["method"] == "regionft_d10")]
    if view == "partition":
        return [row for row in rows
                if row["method"] in ("regionft_d10", "regionft_random")]
    raise ValueError(f"unknown RegionFT analysis view: {view}")


def _view_axis(view, groups):
    if view == "partition":
        return "source", ("gini", "random")
    values = sorted({value for cell in groups.values() for value in cell})
    return ("max_depth" if view == "depth" else "alpha"), values


def write_view_summary(rows, view, output_path):
    """Average repeats and write one RQ2/RQ3 summary."""
    if view not in VIEW_FILENAMES:
        raise ValueError(f"unknown RegionFT analysis view: {view}")

    groups = _group_views(rows)[view]
    axis_name, axis_values = _view_axis(view, groups)
    columns = ["dataset", "model", "protected", axis_name, "repeats",
               *VALUE_COLUMNS]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(columns)
        for (dataset, model, protected), results in groups.items():
            for axis_value in axis_values:
                if axis_value not in results:
                    continue
                values = results[axis_value]
                writer.writerow([
                    dataset, model, protected, axis_value, values["repeats"],
                    *(values[column] for column in VALUE_COLUMNS),
                ])
    return output_path


def _write_long(path, rows, header):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return path


def analyze_long_results(long_paths, view, summary_path, long_output=None):
    """Generate one readable ablation view from per-run CSV files."""
    rows, header = read_sensitivity_rows(long_paths)
    selected = select_view_rows(rows, view)
    if not selected or not header:
        raise ValueError(f"no rows were found for the RegionFT {view} view")

    written = []
    if long_output is not None:
        written.append(_write_long(long_output, selected, header))
    written.append(write_view_summary(selected, view, summary_path))
    return written


def summarize_views(out_root, views=("depth", "alpha", "partition")):
    analysis_dir = Path(out_root) / "analysis"
    paths = sorted(analysis_dir.glob("sensitivity_long*.csv"))
    rows, _ = read_sensitivity_rows(paths)
    return [
        write_view_summary(rows, view, analysis_dir / VIEW_FILENAMES[view])
        for view in views
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Summarize the RegionFT depth, alpha, and partition ablations.")
    parser.add_argument("--out-root",
                        default=str(RQ2_RQ3_RESULTS_DIR),
                        help="RQ2/RQ3 result directory")
    parser.add_argument("--view", choices=(*VIEW_FILENAMES, "all"), default="all")
    args = parser.parse_args()

    views = tuple(VIEW_FILENAMES) if args.view == "all" else (args.view,)
    for path in summarize_views(args.out_root, views):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
