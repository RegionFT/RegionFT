"""Train the classifiers used by the evaluation."""

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from joblib import dump

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from Experiments.common.datasets import DATASET_FILES
from Experiments.common.paths import CUT_DIR, DATASETS_DIR


TEST_LOG_FILE = CUT_DIR / "test_acc_log.json"


def log_test_performance(dataset_name, model_name, metrics):
    """Update the held-out metrics stored for one trained classifier."""
    if not TEST_LOG_FILE.exists():
        TEST_LOG_FILE.write_text("{}")
    with TEST_LOG_FILE.open() as json_file:
        test_log = json.load(json_file)
    test_log.setdefault(dataset_name, {})
    test_log[dataset_name][model_name] = metrics
    with TEST_LOG_FILE.open("w") as json_file:
        json.dump(test_log, json_file, indent=4)


def _build_classifier(model_name, random_seed):
    if model_name == "DecTree":
        return DecisionTreeClassifier(
            criterion="gini", splitter="best", max_depth=None,
            random_state=random_seed,
        )
    if model_name == "RanForest":
        return RandomForestClassifier(
            n_estimators=50, criterion="gini", random_state=random_seed,
        )
    if model_name == "LogReg":
        return LogisticRegression(penalty="l2", random_state=random_seed)
    if model_name == "NB":
        return CategoricalNB(alpha=1.0)
    if model_name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(50, 30, 15, 10, 5),
            activation="relu", solver="adam", learning_rate="adaptive",
            random_state=random_seed,
        )
    if model_name == "Adaboost":
        return AdaBoostClassifier(
            n_estimators=100, algorithm="SAMME", random_state=random_seed,
        )
    if model_name == "GBDT":
        return GradientBoostingClassifier(
            n_estimators=100,
            random_state=random_seed,
        )
    if model_name == "SVM":
        return LinearSVC(penalty="l2", random_state=random_seed)
    raise ValueError(f"unknown classifier: {model_name}")


def train_CuT(
    train_data,
    model_name,
    save_to="CuT.joblib",
    need_read_train_data_from_file=False,
    test_size=0.2,
    random_seed=314159,
):
    """Train and evaluate one classifier under test."""
    if need_read_train_data_from_file:
        filename = DATASET_FILES.get(train_data, f"{train_data}.csv")
        dataframe = pd.read_csv(DATASETS_DIR / filename)
        values = dataframe.values
        dataset_name = train_data
    else:
        values = train_data
        dataset_name = "custom"

    features = values[:, :-1]
    labels = values[:, -1]
    split = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_seed,
    )
    X_train, X_test, y_train, y_test = split

    try:
        classifier = _build_classifier(model_name, random_seed)
    except ValueError:
        print(f"no ML algorithm called {model_name}.")
        return None
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    if hasattr(classifier, "predict_proba"):
        y_score = classifier.predict_proba(X_test)[:, 1]
    else:
        y_score = classifier.decision_function(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "random_seed": random_seed,
    }
    log_test_performance(dataset_name, model_name, metrics)
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    )
    summary = ", ".join(f"{name}={metrics[name]:.4f}" for name in metric_names)
    print(
        f"{dataset_name}, {model_name}: {summary}, "
        f"random_seed={random_seed}"
    )

    if save_to is not None:
        dump(classifier, save_to)
    return classifier


def train_CuTs():
    """Train the four classifier families for all evaluation datasets."""
    for model_name in ("GBDT", "MLP", "LogReg", "DecTree"):
        for dataset_name in ("Adult", "Credit", "Bank", "Lsac"):
            output = CUT_DIR / f"{model_name}{dataset_name}.joblib"
            train_CuT(
                dataset_name,
                model_name,
                save_to=output,
                need_read_train_data_from_file=True,
            )


if __name__ == "__main__":
    train_CuTs()
