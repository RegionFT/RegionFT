from tensorflow import keras
import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import set_random_seed
from numpy.random import seed
from tensorflow.keras.utils import to_categorical



seed(1)
set_random_seed(2)
import sys, os
sys.path.append("..")
# sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
sys.path.append("/data/d/liliquan/fairness/tabular_fairness")
from preprocessing import pre_german_credit

X_train, X_val, y_train, y_val, constraint \
    = pre_german_credit.X_train, pre_german_credit.X_val, pre_german_credit.y_train, pre_german_credit.y_val, pre_german_credit.constraint
# for i in range(len(X_train[0])):
#     a = X_train[:, i]
#     print("index", i, ":", np.unique(a, return_counts=True))
# exit()
def construct_model(frozen_layers, attr):
    in_shape = X_train.shape[1:]
    # input = keras.Input(shape=in_shape)
    layer1 = keras.layers.Dense(50, activation="relu", name="layer1",input_shape=in_shape)
    layer2 = keras.layers.Dense(30, activation="relu", name="layer2")
    layer3 = keras.layers.Dense(15, activation="relu", name="layer3")
    layer4 = keras.layers.Dense(10, activation="relu", name="layer4")
    layer5 = keras.layers.Dense(5, activation="relu", name="layer5")
    # layer6 = keras.layers.Dense(1, activation="sigmoid", name="layer6")
    c = category_map[attr]
    if attr == 'g':
        last_layer = keras.layers.Dense(c, activation="softmax", name='layer_' + attr)
    else:
        last_layer = keras.layers.Dense(c, activation="softmax", name='layer_' + attr)

    layer_lst = [layer1, layer2, layer3, layer4, layer5]
    for layer in layer_lst[0: frozen_layers]:
        layer.trainable = False
    # x = input
    # for i, l in enumerate(layer_lst):
    #     x = l(x)
    # # y_income = layer6(x)
    # y = last_layer(x)
    model = keras.Sequential([layer1, layer2, layer3, layer4, layer5, last_layer])
    # return keras.Model(input, y_race)
    return model

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fine-tune models with protected attributes')
    parser.add_argument('--path', default='models/german_model.h5', help='model_path')
    parser.add_argument('--attr', default='a', help='protected attributes')
    args = parser.parse_args()

    pos_map = { 'a': 9,
                'g': 6,
            }
    category_map = {'a': 4,
                    'g': 4,
               }
    frozen_layers = [1, 2, 3, 4]

    for frozen_layer in frozen_layers:
        model = construct_model(frozen_layer, args.attr)
        model.load_weights(args.path, by_name=True)
        # attrs = args.a.split('&')
        attr = args.attr
        losses = {}
        losses_weights = {}
        metrics = {}
        y_train_labels = {}
        y_val_labels = {}
        last_layer_name = 'layer_' + attr

        if attr == "g":
            losses[last_layer_name] = 'binary_crossentropy'
        else:
            losses[last_layer_name] = 'categorical_crossentropy'
        losses_weights[last_layer_name] = 1.0
        metrics[last_layer_name] = "accuracy"
        if attr == "g":
            y_train_labels[last_layer_name] = to_categorical(X_train[:, pos_map[attr]] - 1,
                                                             num_classes=category_map[attr])
            y_val_labels[last_layer_name] = to_categorical(X_val[:, pos_map[attr]] - 1,
                                                           num_classes=category_map[attr])
        elif attr == "a":
            y_train_labels[last_layer_name] = to_categorical(X_train[:, pos_map[attr]] - 1,
                                                             num_classes=category_map[attr])
            y_val_labels[last_layer_name] = to_categorical(X_val[:, pos_map[attr]] - 1,
                                                           num_classes=category_map[attr])

        # nadam = keras.optimizers.Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        model.compile(loss=losses, loss_weights=losses_weights, optimizer="nadam", metrics=metrics)

        history = model.fit(x=X_train, y=y_train_labels, epochs=60,
                            validation_data=(X_val, y_val_labels))
        # save model.
        root_path = 'models/finetuned_models_protected_attributes/german/'
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        model_name =  root_path + args.attr + '_german_model_' + str(frozen_layer) + "_" + str(round(history.history["val_acc"][-1], 3)) + '.h5'
        keras.models.save_model(model, model_name)
