import numpy as np
import sys

sys.path.append("../")


def lsac_data(
    path="../datasets/lsac",
):
    """
    Prepare the data of dataset LSAC (law school)
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

    input_shape = (None, 11)
    nb_classes = 2

    return X, Y, input_shape, nb_classes


def lsac_predict_data(
    paths=["../datasets/lsac"],
):
    """
    Prepare the data of dataset LSAC (law school)
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
                X.append(L[:11])

    X = np.array(X, dtype=float)

    input_shape = (None, 11)
    nb_classes = 2

    return X, None, input_shape, nb_classes


def lsac_eval_data(
    path="../datasets/lsac",
    protected_index=9,  # sex
):
    """
    Prepare the data of dataset LSAC (law school)
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

    input_shape = (None, 11)
    nb_classes = 2

    return X, Y, input_shape, nb_classes
