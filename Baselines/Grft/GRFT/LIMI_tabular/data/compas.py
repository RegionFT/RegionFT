import numpy as np
import sys

sys.path.append("../")
from preprocessing import pre_compas_scores

def compas_data(
    path='/data/c/liliquan/fairness_testing/Latent_Imitator/LIMI_tabular/my_compas_train.csv',
):
    """
    Prepare the data of dataset German Credit
    :return: X, Y, input shape and number of classes
    """
    # X = []
    # Y = []
    # i = 0

    # with open(path, "r") as ins:
    #     for line in ins:
    #         line = line.strip()
    #         line1 = line.split(',')
    #         if i == 0:
    #             i += 1
    #             continue
    #         # L = map(int, line1[:-1])
    #         L = [int(i) for i in line1[:-1]]
    #         X.append(L)
    #         if int(line1[-1]) == 0:
    #             Y.append([1, 0])
    #         else:
    #             Y.append([0, 1])
    # X = np.array(X, dtype=float)
    # Y = np.array(Y, dtype=float)
    X=pre_compas_scores.X
    Y=pre_compas_scores.y
    input_shape = (None, len(X[0]))
    nb_classes = 1

    return X, Y, input_shape, nb_classes

def compas_predict_data(
    paths=["../datasets/credit_sample"],
):
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
                L = [i for i in line1]
                X.append(L)

    X = np.array(X, dtype=float)

    input_shape = (None, len(X[0]))
    nb_classes = 1

    return X, None, input_shape, nb_classes