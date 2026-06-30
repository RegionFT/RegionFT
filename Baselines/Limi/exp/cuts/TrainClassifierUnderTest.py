import os
from joblib import load, dump
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = f"{parent_directory}/Datasets"


def train_CuT(dataset_name, model_name, save_to="CuT.joblib"):
    # read train data
    dataset_csv = "GermanCredit" if dataset_name == "Credit" else dataset_name
    df = pd.read_csv(f"{dataset_path}/{dataset_csv}.csv")
    data = df.values
    X = data[:, :-1]
    y = data[:, -1]

    # train CuT based on train data
    CuT = None
    if model_name == "DecTree":
        CuT = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None)
        CuT.fit(X, y)
    elif model_name == "RanForest":
        CuT = RandomForestClassifier(n_estimators=50, criterion="gini")
        CuT.fit(X, y)
    elif model_name == "LogReg":
        CuT = LogisticRegression(penalty="l2")
        CuT.fit(X, y)
    elif model_name == "NB":
        CuT = CategoricalNB(alpha=1.0)
        CuT.fit(X, y)
    elif model_name == "MLP":
        CuT = MLPClassifier(hidden_layer_sizes=(64, 32, 16, 8, 4, 2), activation="relu", solver="adam", learning_rate="adaptive")
        CuT.fit(X, y)
    elif model_name == "Adaboost":
        CuT = AdaBoostClassifier(n_estimators=100, algorithm="SAMME")
        CuT.fit(X, y)
    elif model_name == "GBDT":
        CuT = GradientBoostingClassifier(n_estimators=100)
        CuT.fit(X, y)
    elif model_name == "SVM":
        CuT = LinearSVC(penalty="l2")
        CuT.fit(X, y)
    else:
        print(f"no ML algorithm called {model_name}.")

    # save CuT
    dump(CuT, save_to)


def train_CuTs():
    model_name_list = ["LogReg", "RanForest", "DecTree", "MLP", "Adaboost", "GBDT", "SVM"]
    dataset_name_list = ["Adult", "Credit", "Bank"]
    for model_name in model_name_list:
        for dataset_name in dataset_name_list:
            train_CuT(dataset_name, model_name, f"{model_name}{dataset_name}.joblib")


if __name__ == "__main__":
    train_CuTs()