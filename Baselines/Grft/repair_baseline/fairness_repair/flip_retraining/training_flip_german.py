 
"""
This python file constructs and trains the model for Census Income Dataset.
"""


import sys, os
sys.path.append("/data/d/liliquan/fairness/tabular_fairness")
# sys.path.append("..")
# sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from preprocessing import pre_german_credit
import tensorflow as tf
from tensorflow import keras
import itertools
import time
import copy
import tensorflow.keras.backend as KTF
import argparse
import random
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def similar_set(X, num_attribs, protected_attribs, constraint):
    # find all similar inputs corresponding to different combinations of protected attributes with non-protected attributes unchanged
    similar_X = []
    protected_domain = []
    for i in protected_attribs:
        protected_domain = protected_domain + [list(range(constraint[i][0], constraint[i][1]+1))]
    all_combs = np.array(list(itertools.product(*protected_domain)))
    for i, comb in enumerate(all_combs):
        X_new = copy.deepcopy(X)
        for a, c in zip(protected_attribs, comb):
            X_new[:, a] = c
        similar_X.append(X_new)
    return similar_X
# create and train a six-layer neural network for the binary classification task
# for benchmark in ['a','g','r','a&g','a&r','r&g']:
for benchmark in ['a','g','g&a']:
    model = keras.Sequential([
        keras.layers.Dense(50, activation="relu", input_shape=pre_german_credit.X_train.shape[1:]),
        keras.layers.Dense(30, activation="relu"),
        keras.layers.Dense(15, activation="relu"),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(5, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    print(model.summary())
    model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])

    # uncomment for training
    """
    history = model.fit(pre_census_income.X_train, pre_census_income.y_train, epochs=30, validation_data=(pre_census_income.X_val, pre_census_income.y_val))
    model.evaluate(pre_census_income.X_test, pre_census_income.y_test) # 84.32% accuracy
    model.save("models/models_from_tests/adult_model.h5")
    """
    pos_map = { 'a': [9],
                'g': [6],
                'g&a': [6, 9],
                }
    # paras_map = {'a': [-0.3, 0.15],
    #             'g': [-0.6, 0.1],
    #             'r': [-0.95, 0.8],
    #             'a&r': [-0.35, 0.25],
    #             'a&g': [-0.3, 0.25],
    #             'r&g': [-0.9, 0.8],
    # }
    X_train = pre_german_credit.X_train
    y_train = pre_german_credit.y_train
    constraint = pre_german_credit.constraint
    num_attribs = len(X_train[0])
    protected_attribs = pos_map[benchmark]
    similar_inputs = similar_set(X_train, num_attribs, protected_attribs, constraint)
    X_train=similar_inputs[0]
    # print(len)
    print("******")
    print(y_train)
    print("******")
    print(len(similar_inputs))

    X_train = np.concatenate(similar_inputs, axis=0)
    similar_labels=[]
    for i in range(0,len(similar_inputs)):
        print(i)
    #     # X_train = np.vstack((X_train,similar_inputs[i]))
    #     # y_train= np.hstack((y_train,y_train))
    #     X_train=np.concatenate((X_train, similar_inputs[i]), axis=0)
        
        similar_labels.append(y_train)
    y_train=np.concatenate(similar_labels, axis=0)
    print(len(y_train))
    print(X_train.shape)
    # 打乱特征和标签，保持对应关系
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)

    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    history = model.fit(X_shuffled, y_shuffled, epochs=30, validation_data=(pre_german_credit.X_val, pre_german_credit.y_val))
    model.evaluate(pre_german_credit.X_test, pre_german_credit.y_test) # 84.32% accuracy
    model.save("models/models_flip/german_model_"+benchmark+".h5")
    # The precision rate is  0.7338425381903643 , the recall rate is  0.5454148471615721 , and the F1 score is  0.625751503006012