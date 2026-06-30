"""Classifier-under-test adapter and model loading helpers."""

from pathlib import Path

from joblib import load

from Experiments.common.datasets import load_dataset
from Experiments.common.paths import CUT_DIR


class BlackBoxModel:
    """Expose a trained estimator through the interface used by the testers."""

    def __init__(self, data_range, model, feature_list):
        self.no_attr = len(data_range)
        self.data_range = data_range
        self.model = model
        self.feature_list = feature_list

    def predict(self, inputs):
        integer_inputs = [
            [int(item) for item in row]
            for row in inputs
        ]
        return self.model.predict(integer_inputs)

    def predict_proba(self, inputs):
        integer_inputs = [
            [int(item) for item in row]
            for row in inputs
        ]
        return self.model.predict_proba(integer_inputs)


def load_cut(dataset_name, model_name, cut_dir=None):
    """Load a classifier under test together with its source dataframe."""
    data_range, dataframe = load_dataset(dataset_name)
    model_path = Path(cut_dir or CUT_DIR) / f"{model_name}{dataset_name}.joblib"
    estimator = load(model_path)
    return (
        BlackBoxModel(
            data_range,
            estimator,
            feature_list=dataframe.columns.tolist(),
        ),
        dataframe,
    )
