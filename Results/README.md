# Results

This directory contains two types of artifacts: a complete single-run example
and the retained results from the experiments reported in the paper.

## Runner Example

[`runner_examples/run_001/`](runner_examples/run_001) contains the complete
output of the 60-second RegionFT example described in the main
[`README`](../README.md#run-a-single-experiment). It includes the run log,
discovered IDI pairs, and the learned partition.

This example illustrates the output structure and is not part of the paper
evaluation.

## Main Experiment Results

The results used in the paper are retained as per-run JSON logs and readable
CSV files.

### Experiment Logs

- [`main_experiments/baseline_comparison/`](main_experiments/baseline_comparison)
  contains the RQ1 logs.
- [`main_experiments/regionft_ablation/`](main_experiments/regionft_ablation)
  contains the RQ2 and RQ3 logs.

The original runs also produced large IDI arrays, partition files, and
coverage-cell archives. These raw outputs total approximately 500 GB and are
omitted, while the metrics extracted from them are retained in the `long` CSV
files. Therefore, the complete per-run metrics cannot be recomputed from the
JSON logs alone.

### Result CSVs

| Evaluation | Per-run results | Aggregated results |
|---|---|---|
| RQ1 baseline comparison | [`RQ1_baseline_long.csv`](RQ1_baseline_long.csv) | [`RQ1_baseline_summary.csv`](RQ1_baseline_summary.csv), [`RQ1_significance.csv`](RQ1_significance.csv) |
| RQ2 partition rule | [`RQ2_partition_rule_long.csv`](RQ2_partition_rule_long.csv) | [`RQ2_partition_rule_summary.csv`](RQ2_partition_rule_summary.csv) |
| RQ2 maximum depth | [`RQ2_max_depth_long.csv`](RQ2_max_depth_long.csv) | [`RQ2_max_depth_summary.csv`](RQ2_max_depth_summary.csv) |
| RQ3 sampling exponent | [`RQ3_alpha_long.csv`](RQ3_alpha_long.csv) | [`RQ3_alpha_summary.csv`](RQ3_alpha_summary.csv) |

A `long` file contains one row per run. A `summary` file groups runs by task
and experimental setting and averages the five repeats. `RQ1_significance.csv`
compares RegionFT with each baseline across the 32 tasks.

### Key Metrics

The CSV files use the following key metric names; the fields included vary by
file.

| Field | Meaning |
|---|---|
| `n_idi` | Number of discovered IDIs. |
| `n_test` | Number of evaluated test cases. |
| `prec` | IDI success rate (`n_idi / n_test`), not classifier precision. |
| `covK` | Number of occupied evaluation-grid cells at granularity `K`. |
| `divN` | Mean pairwise Hamming distance among up to `N` IDIs. |

RQ1 reports whole-run results, while RQ2 and RQ3 report the RegionFT
generation phase.
