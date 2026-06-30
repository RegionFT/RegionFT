import numpy as np
import sys

sys.path.append("../")


def compas_data(
    path="../datasets/compas",
):
    """
    Prepare the data of dataset COMPAS
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open(path, "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(",")
            if i == 0:
                i += 1
                continue
            L = [int(i) for i in line1[:-1]]
            X.append(L)
            if int(line1[-1]) == 0:
                Y.append([1, 0])
            else:
                Y.append([0, 1])
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    input_shape = (None, 6)
    nb_classes = 2

    return X, Y, input_shape, nb_classes


def compas_predict_data(
    paths=["../datasets/compas"],
):
    """
    Prepare the data of dataset COMPAS
    there is no label of the dataset
    :return: X, Y, input shape and number of classes
    """
    X = []

    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        with open(path, "r") as ins:
            i = 0
            for line in ins:
                line = line.strip()
                line1 = line.split(",")
                if i == 0:
                    i += 1
                    continue
                L = [int(i) for i in line1]
                X.append(L[:6])

    X = np.array(X, dtype=float)

    input_shape = (None, 6)
    nb_classes = 2

    return X, None, input_shape, nb_classes


def compas_eval_data(
    path="../datasets/compas",
    protected_index=4,  # sex
):
    """
    Prepare the data of dataset COMPAS
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open(path, "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(",")
            if i == 0:
                i += 1
                continue
            L = [int(i) for i in line1[:-1]]
            X.append(L)
            Y.append([int(line1[-1]), int(L[protected_index])])
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    input_shape = (None, 6)
    nb_classes = 2

    return X, Y, input_shape, nb_classes
