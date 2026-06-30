# Experiments

This directory contains the datasets, trained classifiers, and shared code used
by the main experiment scripts.

## Contents

| Path | Contents |
|---|---|
| [`Datasets/`](Datasets) | Preprocessed datasets and protected attributes. |
| [`CuT/`](CuT) | Trained classifiers under test and the training script. |
| [`common/`](common) | Shared experiment execution and result-extraction code. |
| [`scripts/`](scripts) | Result analysis and partition-calibration code. |

## Run the Main Experiments

Run both scripts from the repository root:

```bash
python run_RQ1_baseline_comparison.py --processes 64
python run_RQ2_RQ3_ablation.py --processes 64
```

| Script | Evaluation | Default execution |
|---|---|---|
| `run_RQ1_baseline_comparison.py` | RQ1 baseline comparison | 32 tasks × 6 methods × 5 repeats; 1,200 seconds per run |
| `run_RQ2_RQ3_ablation.py` | RQ2/RQ3 ablations | 32 tasks × 16 variants × 5 repeats; 5 million generation draws per run |

Both scripts execute each job through [`runner.py`](../runner.py). Use `--help`
to select tasks, methods or variants, budgets, repeats, and worker processes.

## Results

New executions are stored in `Results/RQ1_baseline_comparison/` and
`Results/RQ2_RQ3_ablation/`.

After both experiments finish, generate the final RQ `long`, `summary`, and
`significance` CSV files with:

```bash
python analyze_results.py --from-experiments Results
```

See [`Results/README.md`](../Results/README.md) for the retained experiment
results and metric definitions.
