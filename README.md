# RegionFT

This is the anonymized code artifact for ICSE 2027 submission #1447,
**“Region-Guided Fairness Testing via Adaptive Partitioning.”** RegionFT tests
the individual fairness of black-box classifiers by partitioning their input
space and guiding test generation toward regions with higher estimated
discrimination risk.

Beyond implementing RegionFT, this artifact integrates representative black-box
individual fairness testing methods through a common `runner.py` interface,
enabling researchers to reproduce and compare them more easily.

## Setup

The project uses Python 3.9. Conda is recommended:

```bash
conda env create -f environment.yml
conda activate env_regionft
```

A standard virtual environment can also be used:

```bash
python3.9 -m venv env_regionft
source env_regionft/bin/activate
pip install -r requirements.txt
```

Run the following commands from the repository root.

## Run a Single Experiment

`runner.py` runs one combination of dataset, protected attribute, classifier,
and testing method. It supports the proposed RegionFT method and the
representative methods AFT, ExpGA, GRFT, LIMI, SG, Themis, VBT, and VBT-X. To
view all options, run:

```bash
python runner.py --help
```

The following command runs RegionFT for 60 seconds on the Adult/GBDT
classifier with `sex` as the protected attribute:

```bash
python runner.py \
  --dataset Adult \
  --protected sex \
  --model GBDT \
  --method regionft \
  --runtime 60 \
  --show-log
```

When `--output` is omitted, `runner.py` selects the next unused example
directory:

```text
Results/runner_examples/run_001/regionft/
├── log/*.json
├── disc/*.npy
└── partition/*.csv
```

The output contains:

- `log/*.json`: configuration, metrics, and timing.
- `disc/*.npy`: discovered IDI pairs.
- `partition/*.csv`: learned regions and their statistics.

A completed example is retained in
[`Results/runner_examples/run_001`](Results/runner_examples/run_001).

## Experiment Scripts

The following scripts run individual configurations, execute the main RQ
experiments, and analyze their results.

| Script | Purpose | Output |
|---|---|---|
| [`runner.py`](runner.py) | Run one experiment configuration. | `Results/runner_examples/run_001`, `run_002`, ... |
| [`run_RQ1_baseline_comparison.py`](run_RQ1_baseline_comparison.py) | Run the RQ1 comparison of RegionFT and five baselines. | `Results/RQ1_baseline_comparison/` |
| [`run_RQ2_RQ3_ablation.py`](run_RQ2_RQ3_ablation.py) | Run the RQ2/RQ3 partition and sampling ablations. | `Results/RQ2_RQ3_ablation/` |
| [`analyze_results.py`](analyze_results.py) | Generate readable RQ CSVs from per-run results. | `Results/` by default |

See [`Experiments/README.md`](Experiments/README.md) for batch execution.

## Experiment Results

The logs and readable CSVs used in the evaluation are retained under
[`Results/`](Results). Logs from the main RQ1–RQ3 experiments are stored in
[`Results/main_experiments/`](Results/main_experiments), while the corresponding
per-run metrics and summary tables are stored directly under `Results/`.

See [`Results/README.md`](Results/README.md) for the file organization and
metric definitions.

## Project Layout

The repository is organized into the following top-level directories.

| Directory | Contents |
|---|---|
| [`RegionFT/`](RegionFT) | Implementation of the proposed RegionFT method. |
| [`Baselines/`](Baselines) | Baseline implementations and their sources. |
| [`Experiments/`](Experiments) | Datasets, trained classifiers, and shared experiment code. |
| [`Results/`](Results) | Retained logs, readable CSVs, and a complete runner example. |
