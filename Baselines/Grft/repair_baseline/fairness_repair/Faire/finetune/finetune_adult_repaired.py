import sys, os
sys.path.append("/data/d/liliquan/fairness/tabular_fairness")
# sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

from tensorflow import keras
import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import set_random_seed
from numpy.random import seed
from tensorflow.keras.utils import to_categorical
from preprocessing import pre_census_income
from scalelayer import  ScaleLayer
from tensorflow.keras.models import Model
from tensorflow.keras import activations

seed(1)
set_random_seed(2)


X_train, X_val, y_train, y_val, constraint = pre_census_income.X_train, \
    pre_census_income.X_val, pre_census_income.y_train, pre_census_income.y_val, pre_census_income.constraint

def construct_model(frozen_layers, attr):
    in_shape = inner_output_train.shape[1:]
 
    # input = keras.Input(shape=in_shape)
    layer1 = keras.layers.Dense(30, activation="relu", name="layer1",input_shape=in_shape)
    layer2 = keras.layers.Dense(20, activation="relu", name="layer2")
    layer3 = keras.layers.Dense(15, activation="relu", name="layer3")
    layer4 = keras.layers.Dense(15, activation="relu", name="layer4")
    layer5 = keras.layers.Dense(10, activation="relu", name="layer5")
    # layer6 = keras.layers.Dense(1, activation="sigmoid", name="layer6")
    c = category_map[attr]
    if attr == 'g':
        last_layer = keras.layers.Dense(c, activation="sigmoid", name='layer_' + attr)
    else:
        last_layer = keras.layers.Dense(c, activation="softmax", name='layer_' + attr)
        
    act = keras.layers.Activation(activations.relu)    
    layer_lst = [layer1, layer2, layer3, layer4, layer5]
    
    layer_lst = layer_lst[frozen_layers:]
    
    # x = act(input)
    # for i, l in enumerate(layer_lst):
    #     x = l(x)
    # # y_income = layer6(x)
    # y = last_layer(x)
    model = keras.Sequential([*layer_lst, last_layer])
    # return keras.Model(input, y_race)
    return model


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fine-tune models with protected attributes')
#     parser.add_argument('--path', default='models/retrained_models_EIDIG/adult_EIDIG_INF_retrained_model.h5', help='model_path')
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
    frozen_layers = [1, 2, 3, 4]
#     frozen_layers = [4]
    
    models_map = {
        'a': "models/adult_model.h5",
        'r': "models/adult_model.h5",
        'g': "models/adult_model.h5",
            
#         'a': "models/gated_models/adult_a_gated_4_0.3_0.2_p-0.3_p0.15.h5",
#         'r': "models/gated_models/adult_g_gated_4_0.3_0.2_p-0.6_p0.1.h5",
#         'g': "models/gated_models/adult_r_gated_4_0.3_0.2_p-0.95_p0.8.h5",
#         'a&r': "models/adult_a&r_gated_4_0.3_0.2_p-0.4_p0.5.h5",
#         'a&g': "models/adult_a&g_gated_4_0.3_0.2_p-0.3_p0.2.h5",
#         'r&g': "models/adult_r&g_gated_4_0.3_0.2_p-0.9_p0.8.h5",
    }

    for frozen_layer in frozen_layers:
        model_path = models_map[args.attr]
        base_model = keras.models.load_model(model_path, custom_objects={'ScaleLayer': ScaleLayer})
#         layer_name = "scale_layer_" + str(frozen_layer+4)
        layer_name = "layer" + str(frozen_layer)
        inner_model = Model(base_model.input, base_model.get_layer(layer_name).output)                                 
        inner_output_train = inner_model.predict(pre_census_income.X_train)
        inner_output_val = inner_model.predict(pre_census_income.X_val)
        
        model = construct_model(frozen_layer, args.attr)
        
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

        # nadam = keras.optimizers.Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        model.compile(loss=losses, loss_weights=losses_weights, optimizer="nadam", metrics=metrics)
#         print(len(y_val_labels[last_layer_name]))
        history = model.fit(x=inner_output_train, y=y_train_labels, epochs=30,
                            validation_data=(inner_output_val, y_val_labels))
        
        root_path = 'models/finetuned_models_protected_attributes/adult/'
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        # save model.
        model_name = root_path + args.attr + '_adult_model_' + str(frozen_layer) + "_" + str(round(history.history["val_acc"][-1], 3)) + '.h5'
        keras.models.save_model(model, model_name)
