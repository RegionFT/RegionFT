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
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

from preprocessing import pre_census_income
X_train, X_val, y_train, y_val, constraint = pre_census_income.X_train, \
    pre_census_income.X_val, pre_census_income.y_train, pre_census_income.y_val, pre_census_income.constraint


def construct_model(frozen_layers, adv):
    in_shape = X_train.shape[1:]
    input = keras.Input(shape=in_shape)
    layer1 = keras.layers.Dense(30, activation="relu", name="layer1")
    layer2 = keras.layers.Dense(20, activation="relu", name="layer2")
    layer3 = keras.layers.Dense(15, activation="relu", name="layer3")
    layer4 = keras.layers.Dense(15, activation="relu", name="layer4")
    layer5 = keras.layers.Dense(10, activation="relu", name="layer5")
    layer6 = keras.layers.Dense(1, activation="sigmoid", name="layer6")
    last_layers = []
    if adv:
        for attr in attrs:
            c = category_map[attr]
            if attr == 'g':
                last_layer = keras.layers.Dense(c, activation="sigmoid", name='layer_' + attr)
            else:
                last_layer = keras.layers.Dense(c, activation="softmax", name='layer_' + attr)
            last_layers.append(last_layer)
            
    layer_lst = [layer1, layer2, layer3, layer4, layer5]

    x = input
    for i, l in enumerate(layer_lst):
        x = l(x)
        
    y_income = layer6(x)
    model = keras.Model(input, [y_income])
    
    y_advs = []
    
    if adv:
        for index, attr in enumerate(attrs):
            y_adv = last_layers[index](x)
            y_advs.append(y_adv)
        
        model = keras.Model(input, [y_income, *y_advs])
        
    return model

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fine-tune models with protected attributes')
    parser.add_argument('--path', default='models/retrained_model_EIDIG/adult_EIDIG_INF_retrained_model.h5', help='model_path')
    parser.add_argument('--attr', default='a', help='protected attributes')
    args = parser.parse_args()

    pos_map = { 'a': 0,
            'r': 6,
            'g': 7,
            }
    category_map = {'a': 4,
               'r': 5,
               'g': 1,
               }
    
    frozen_layer = 0
    attrs = args.attr.split('&')
    model = construct_model(frozen_layer, adv=True)
    model.summary()
#         model.load_weights(args.path, by_name=True)
    losses = {}
    losses_weights = {}
    metrics = {}
    y_train_labels = {}
    y_val_labels = {}
    for attr in attrs:
        last_layer_name = 'layer_' + attr
        losses_weights[last_layer_name] = -1.0
        metrics[last_layer_name] = "accuracy"
        if attr == "g":
            losses[last_layer_name] = 'mean_squared_error'
        else:
            losses[last_layer_name] = 'mean_squared_error'      
        if attr == "g":
            y_train_labels[last_layer_name] = X_train[:, pos_map[attr]]
            y_val_labels[last_layer_name] = X_val[:, pos_map[attr]]
        elif attr == "a":
            y_train_labels[last_layer_name] = to_categorical(X_train[:, pos_map[attr]]-1,
                                                             num_classes=category_map[attr])
            y_val_labels[last_layer_name] = to_categorical(X_val[:, pos_map[attr]]-1,
                                                               num_classes=category_map[attr])
        elif attr == "r":
            y_train_labels[last_layer_name] = to_categorical(X_train[:, pos_map[attr]],
                                                             num_classes=category_map[attr])
            y_val_labels[last_layer_name] = to_categorical(X_val[:, pos_map[attr]],
                                                           num_classes=category_map[attr])
            
    losses["layer6"] = 'binary_crossentropy'
    
    losses_weights["layer6"] = 1.0
    
    metrics["layer6"] = "accuracy"

    y_train_labels['layer6'] = y_train
    y_val_labels['layer6'] = y_val

    # nadam = keras.optimizers.Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(loss=losses, loss_weights=losses_weights, optimizer="nadam", metrics=metrics)
    
    
    history = model.fit(x=X_train, y=y_train_labels, epochs=10, validation_data=(X_val, y_val_labels))

    # save model.
    file_path = 'models/retrained_adv/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    model_name = (file_path + args.attr + '_adult_multi_model_' + str(frozen_layer) + '.h5')
    tf.keras.models.save_model(model, model_name)

    saved_model = construct_model(frozen_layer, adv=False)
    saved_model.load_weights(model_name, by_name=True)
    model_name = (file_path + args.attr + '_adult_model_' + str(frozen_layer) + '.h5')
    tf.keras.models.save_model(saved_model, model_name)
    
    print("saved!")