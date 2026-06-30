"""Repository paths resolved independently of the current working directory."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "Experiments"

DATASETS_DIR = EXPERIMENTS_DIR / "Datasets"
CUT_DIR = EXPERIMENTS_DIR / "CuT"

ARCHIVED_RESULTS_DIR = REPO_ROOT / "Results"
RQ1_RESULTS_DIR = ARCHIVED_RESULTS_DIR / "RQ1_baseline_comparison"
RQ2_RQ3_RESULTS_DIR = ARCHIVED_RESULTS_DIR / "RQ2_RQ3_ablation"
