"""Dataset metadata and loading helpers shared by the experiment workflow."""

from pathlib import Path

import pandas as pd

from Experiments.common.paths import DATASETS_DIR


DATASET_FILES = {
    "Adult": "Adult.csv",
    "Credit": "GermanCredit.csv",
    "Bank": "Bank.csv",
    "Lsac": "Lsac.csv",
}

PROTECTED_ATTRIBUTES = {
    "Adult": [("sex", 8), ("race", 7), ("age", 0)],
    "Credit": [("sex", 8), ("age", 12)],
    "Bank": [("age", 0)],
    "Lsac": [("sex", 9), ("race", 10)],
}


def read_data_range(path):
    """Return closed feature ranges and the full dataset dataframe."""
    dataframe = pd.read_csv(Path(path))
    data_range = [
        [dataframe.iloc[:, index].min(), dataframe.iloc[:, index].max()]
        for index in range(dataframe.shape[1] - 1)
    ]
    return data_range, dataframe


def load_dataset(dataset_name):
    """Load one evaluation dataset by its public experiment name."""
    return read_data_range(DATASETS_DIR / DATASET_FILES[dataset_name])
