"""Summarize the RQ1 baseline-comparison results."""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

from Experiments.common.datasets import PROTECTED_ATTRIBUTES
from run_RQ1_baseline_comparison import METHODS, MODELS


SUMMARY_STD_METRICS = ("n_idi", "cov2", "cov3", "cov4", "cov5", "div5000")
SIGNIFICANCE_METRICS = (
    ("n_idi", "n_idi", 2),
    ("coverage_k3", "cov3", 2),
    ("diversity_n1000", "div1000", 4),
)
SIGNIFICANCE_BASELINES = (
    ("aft", "AFT"),
    ("expga", "ExpGA"),
    ("grft", "GRFT"),
    ("limi", "LIMI"),
    ("sg", "SG"),
)
SIGNIFICANCE_COLUMNS = (
    "metric", "baseline", "n_tasks", "regionft_mean", "baseline_mean",
    "wins", "ties", "losses", "A12", "wilcoxon_p",
)
BASE_COLUMNS = ("run_id", "dataset", "model", "protected", "method", "repeat")

_DATASET_RANK = {dataset: index for index, dataset in enumerate(PROTECTED_ATTRIBUTES)}
_MODEL_RANK = {model: index for index, model in enumerate(MODELS)}
_METHOD_RANK = {method: index for index, method in enumerate(METHODS)}


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _protected_rank(dataset, protected):
    names = [name for name, _ in PROTECTED_ATTRIBUTES.get(dataset, ())]
    return names.index(protected) if protected in names else len(names)


def _row_key(row):
    try:
        repeat = int(row.get("repeat", 0))
    except (TypeError, ValueError):
        repeat = 0
    return (
        _DATASET_RANK.get(row["dataset"], 99),
        _MODEL_RANK.get(row["model"], 99),
        _protected_rank(row["dataset"], row["protected"]),
        _METHOD_RANK.get(row["method"], 99),
        row["method"],
        repeat,
    )


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


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


def _number(value):
    if value in (None, "", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summary_value(column, mean):
    if mean is None:
        return ""
    if column in ("n_idi", "n_test") or column.startswith("cov"):
        return round(mean, 2)
    if column.startswith("div"):
        return round(mean, 4)
    if column == "n_regions":
        return round(mean, 1)
    return round(mean, 2)


def aggregate_summary(rows, header, include_std=False):
    """Average repeats for each dataset, model, protected attribute, and method."""
    value_columns = [
        column for column in (header or ())
        if column not in BASE_COLUMNS
    ]
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["model"],
            row["protected"],
            row["method"],
        )
        groups[key].append(row)

    summary = []
    for (dataset, model, protected, method), members in groups.items():
        result = {
            "dataset": dataset,
            "model": model,
            "protected": protected,
            "method": method,
            "repeats": len(members),
        }
        numeric_values = {}
        for column in value_columns:
            values = [_number(row.get(column)) for row in members]
            numeric_values[column] = [
                value for value in values if value is not None
            ]
        means = {
            column: _mean(values)
            for column, values in numeric_values.items()
        }
        result.update({
            column: _summary_value(column, means[column])
            for column in value_columns
        })

        # Precision is computed from the aggregate IDI and test counts.
        if "n_idi" in value_columns and "n_test" in value_columns:
            n_idi = means.get("n_idi")
            n_test = means.get("n_test")
            result["prec"] = (
                round(n_idi / n_test, 6)
                if n_idi is not None and n_test
                else 0.0
            )

        if include_std:
            for column in SUMMARY_STD_METRICS:
                if column not in value_columns:
                    continue
                values = numeric_values[column]
                result[f"{column}_std"] = (
                    round(statistics.stdev(values), 4)
                    if len(values) > 1
                    else 0.0
                )
        summary.append(result)

    summary.sort(key=_row_key)
    columns = ["dataset", "model", "protected", "method", "repeats"]
    if "n_idi" in value_columns:
        columns.extend(("n_idi", "n_test", "prec"))
    columns.extend(
        column for column in value_columns
        if column not in ("n_idi", "n_test")
    )
    if include_std:
        columns.extend(
            f"{column}_std"
            for column in SUMMARY_STD_METRICS
            if column in value_columns
        )
    return summary, columns


def _summary_by_task(summary_rows):
    by_task = defaultdict(dict)
    for row in summary_rows:
        task = (row["dataset"], row["model"], row["protected"])
        by_task[task][row["method"]] = row
    return by_task


def _paired_values(tasks, baseline, column):
    pairs = []
    for methods in tasks.values():
        if "regionft" not in methods or baseline not in methods:
            continue
        regionft_value = _number(methods["regionft"].get(column))
        baseline_value = _number(methods[baseline].get(column))
        if regionft_value is not None and baseline_value is not None:
            pairs.append((regionft_value, baseline_value))
    return pairs


def build_significance(summary_rows):
    """Compute paired wins, A12, and Wilcoxon tests against each baseline."""
    from scipy.stats import wilcoxon

    tasks = _summary_by_task(summary_rows)
    results = []
    for metric_name, column, digits in SIGNIFICANCE_METRICS:
        for baseline, display_name in SIGNIFICANCE_BASELINES:
            pairs = _paired_values(tasks, baseline, column)
            if not pairs:
                continue

            regionft_values = [pair[0] for pair in pairs]
            baseline_values = [pair[1] for pair in pairs]
            wins = sum(left > right for left, right in pairs)
            ties = sum(left == right for left, right in pairs)
            losses = len(pairs) - wins - ties
            effect_size = (wins + 0.5 * ties) / len(pairs)
            effect_size = (
                int(effect_size)
                if effect_size.is_integer()
                else round(effect_size, 4)
            )
            differences = [left - right for left, right in pairs]
            p_value = (
                1.0
                if not any(differences)
                else wilcoxon(regionft_values, baseline_values).pvalue
            )
            results.append({
                "metric": metric_name,
                "baseline": display_name,
                "n_tasks": len(pairs),
                "regionft_mean": round(
                    statistics.fmean(regionft_values),
                    digits,
                ),
                "baseline_mean": round(
                    statistics.fmean(baseline_values),
                    digits,
                ),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "A12": effect_size,
                "wilcoxon_p": f"{p_value:.3e}",
            })
    return results


def analyze_long_results(
    long_paths,
    summary_path,
    significance_path,
    long_output=None,
):
    """Generate the RQ1 long, summary, and significance tables."""
    rows, header = read_sensitivity_rows(long_paths)
    if not rows or not header:
        raise ValueError("no baseline-comparison long rows were found")

    written = []
    if long_output is not None:
        long_output = Path(long_output)
        long_output.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(long_output, sorted(rows, key=_row_key), header)
        written.append(long_output)

    summary_rows, summary_columns = aggregate_summary(
        rows,
        header,
        include_std=True,
    )
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(summary_path, summary_rows, summary_columns)
    written.append(summary_path)

    significance_path = Path(significance_path)
    significance_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(
        significance_path,
        build_significance(summary_rows),
        SIGNIFICANCE_COLUMNS,
    )
    written.append(significance_path)
    return written
