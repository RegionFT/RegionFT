"""Adapters for executing the fairness-testing methods."""

import os
import time
from pathlib import Path
from typing import Any, NamedTuple, Optional, Tuple

from Experiments.common.paths import CUT_DIR, REPO_ROOT
from RegionFT.execution import run_regionft


EXPGA_THRESHOLD = {"Adult": 7, "Credit": 14, "Bank": 10, "Lsac": 7}
LIMI_DATASET_NAMES = {
    "Adult": "census",
    "Credit": "credit",
    "Bank": "bank",
    "Lsac": "lsac",
}


class MethodRun(NamedTuple):
    """Inputs shared by the method-specific adapters."""

    method: str
    cut: Any
    dataset_name: str
    model_name: str
    protected_name: str
    protected_index: int
    original_data: Any
    runtime: int
    label: Tuple[str, int]
    disc_dir: Path
    test_dir: Optional[Path]
    partition_dir: Optional[Path]
    show_logging: bool
    regionft_config: Optional[dict]
    cut_dir: Optional[Path] = None


def is_regionft(method):
    """Return whether a method uses the RegionFT execution path."""
    return method == "regionft" or method.startswith("regionft_")


def regionft_variant_overrides(method):
    """Decode a RegionFT ablation name into configuration overrides."""
    if method == "regionft":
        return {}

    suffix = method.removeprefix("regionft_")
    if suffix == "random":
        return {"partition_mode": "random"}
    if suffix == "fixed":
        return {}
    if suffix.startswith("d") and suffix[1:].isdigit():
        return {"max_depth": int(suffix[1:])}
    if suffix.startswith("a") and suffix[1:].replace("p", "").isdigit():
        alpha = float(suffix[1:].replace("p", "."))
        return {"sampling_risk_focus_alpha": alpha}
    raise ValueError(f"unknown RegionFT variant method: {method!r}")


def _save_dirs(run):
    return (
        str(run.disc_dir),
        None if run.test_dir is None else str(run.test_dir),
    )


def _run_aft(run):
    from Baselines import AFT

    disc_dir, test_dir = _save_dirs(run)
    tester = AFT(
        run.cut,
        [run.protected_index],
        no_train_data_sample=5000,
        show_logging=run.show_logging,
    )
    tester.test(
        runtime=run.runtime,
        max_leaf_nodes=1000,
        max_train_data_each_path=10,
        max_sample_each_path=100,
        MaxDiscPathPair=100,
        MaxTry=10000,
        dt_search_mode="random+flip",
        check_type="themis",
        label=run.label,
        disc_save_to=disc_dir,
        test_save_to=test_dir,
    )
    return tester


def _run_vbt(run):
    from Baselines import Vbt, Vbtx

    disc_dir, test_dir = _save_dirs(run)
    tester_class = Vbtx if run.method == "vbtx" else Vbt
    version = "improved" if run.method == "vbtx" else "vbt"
    tester = tester_class(
        run.cut,
        [run.protected_index],
        no_train_data_sample=5000,
        vbtx_ver=version,
        show_logging=run.show_logging,
    )
    tester.test(
        runtime=run.runtime,
        label=run.label,
        disc_save_to=disc_dir,
        test_save_to=test_dir,
    )
    return tester


def _run_themis(run):
    from Baselines import Themis

    disc_dir, test_dir = _save_dirs(run)
    tester = Themis(
        run.cut,
        [run.protected_index],
        show_logging=run.show_logging,
    )
    tester.test(
        runtime=run.runtime,
        max_test=None,
        max_disc=None,
        label=run.label,
        disc_save_to=disc_dir,
        test_save_to=test_dir,
    )
    return tester


def _run_expga(run):
    from Baselines import ExpGA

    disc_dir, test_dir = _save_dirs(run)
    tester = ExpGA(
        run.cut,
        [run.protected_index],
        protected_name=run.protected_name,
        threshold_l=EXPGA_THRESHOLD[run.dataset_name],
        original_data=run.original_data,
        show_logging=run.show_logging,
    )
    tester.xai_fair_testing(
        runtime=run.runtime,
        max_local=None,
        seed_num=None,
        label=run.label,
        disc_save_to=disc_dir,
        test_save_to=test_dir,
    )
    return tester


def _run_sg(run):
    from Baselines import SG

    disc_dir, test_dir = _save_dirs(run)
    tester = SG(
        dataset_name=run.dataset_name,
        black_box_model=run.cut,
        protected_list=[run.protected_index],
        protected_name=run.protected_name,
        original_data=run.original_data,
        show_logging=run.show_logging,
    )
    tester.symbolic_generation(
        limit_test=None,
        limit_seed=100000,
        cluster_num=4,
        runtime=run.runtime,
        label=run.label,
        disc_save_to=disc_dir,
        test_save_to=test_dir,
    )
    return tester


def _run_grft(run):
    from Baselines.Grft import Grft

    disc_dir, test_dir = _save_dirs(run)
    tester = Grft(
        run.cut,
        [run.protected_index],
        original_data=run.original_data,
        show_logging=run.show_logging,
    )
    tester.test(
        runtime=run.runtime,
        label=run.label,
        disc_save_to=disc_dir,
        test_save_to=test_dir,
    )
    return tester


def _run_regionft(run):
    return run_regionft(
        cut=run.cut,
        protected_index=run.protected_index,
        runtime=run.runtime,
        label=run.label,
        disc_dir=run.disc_dir,
        test_dir=run.test_dir,
        partition_dir=run.partition_dir,
        show_logging=run.show_logging,
        config=run.regionft_config,
    )


def _run_limi(run):
    from Baselines import Limi

    limi_dir = REPO_ROOT / "Baselines" / "Limi"
    dataset = LIMI_DATASET_NAMES[run.dataset_name]
    model_path = Path(run.cut_dir or CUT_DIR) / (
        f"{run.model_name}{run.dataset_name}.joblib"
    )

    start_time = time.time()
    tester = Limi(
        black_box_model=None,
        black_box_model_path=str(model_path),
        gan_path=str(
            limi_dir / "exp" / "gans" / dataset / f"{dataset}_gan.pth"
        ),
        dataset_path=str(limi_dir / "datasets" / f"{dataset}_train.csv"),
        num_samples=1_000_000,
        dataset_name=dataset,
        dataset_name_new=run.dataset_name,
        protected_attr_name=run.protected_name,
        protected_attr_index=run.protected_index,
        model_name=run.model_name,
        runtime=run.runtime,
        loop=run.label[1],
        step=0.3,
        cwd=str(limi_dir),
        show_logging=run.show_logging,
        disc_save_to=str(run.disc_dir) + os.sep,
        test_save_to=str(run.test_dir) + os.sep,
    )
    tester.test()
    tester.real_time_consumed = time.time() - start_time
    return tester


def run_method(run):
    """Execute one method-specific adapter."""
    if run.method == "aft":
        return _run_aft(run)
    if run.method in ("vbtx", "vbt"):
        return _run_vbt(run)
    if run.method == "themis":
        return _run_themis(run)
    if run.method == "expga":
        return _run_expga(run)
    if run.method == "sg":
        return _run_sg(run)
    if run.method == "grft":
        return _run_grft(run)
    if is_regionft(run.method):
        return _run_regionft(run)
    if run.method == "limi":
        return _run_limi(run)

    print(f"No method called {run.method}")
    return None
