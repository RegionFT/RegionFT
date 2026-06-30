import sys, os
sys.path.append("/data/d/liliquan/fairness/tabular_fairness")
# sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

from tensorflow import keras
import os
import joblib
import numpy as np
from explain import  get_relevance, get_critical_neurons
import tensorflow as tf
# from tensorflow import set_random_seed
from scalelayer import  ScaleLayer
from numpy.random import seed
import itertools
import time
import copy
from preprocessing import pre_census_income
import tensorflow.keras.backend as KTF
import argparse
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, dense_len, min=-1, max=1, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
        tf.keras.constraints.MinMaxNorm()
        self.scale = K.variable([[1. for x in range(dense_len)]], name='ffff',
                                constraint=lambda t: tf.clip_by_value(t, min, max))
        self.dense_len = dense_len
    def call(self, inputs, **kwargs):
        m = inputs * self.scale
        return m
    def get_config(self):
        config = {'dense_len': self.dense_len}
        base_config = super(ScaleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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

def get_repaired_num(newdata_res):
    # identify whether the instance is discriminatory w.r.t. the model
    # print(x.shape)
    # print(X_train[0].shape)
    # y_pred = (model(tf.constant([X_train[0]])) > 0.5)
    l = len(newdata_res)
    for i in range(l-1):
        tmp_acc = (newdata_res[i] == newdata_res[i + 1]) * 1
        if i == 0:
            acc = tmp_acc
        else:
            acc += tmp_acc
    return np.sum(np.where(acc == l-1, True, False))
        
        
        
if __name__ == '__main__': 

    pos_map = { 'a': [0],
            'r': [6],
            'g': [7],
            'a&r': [0, 6],
            'a&g': [0, 7],
            'r&g': [6, 7]
            }
    dataset=pre_census_income
    m ="models/retrained_models/adult_EIDIG_INF_retrained_model.h5"
    new_model = keras.models.load_model(m, custom_objects={'ScaleLayer': ScaleLayer})

    for attr in ["a","r","g","a&r","a&g","r&g"]:
        data_name = f"discriminatory_data/C-{attr}_ids_EIDIG_INF_1.npy"

        dis_data = np.load(data_name)
        print("======",len(dis_data))
        num_attribs = len(dis_data[0])
        protected_attribs = pos_map[attr]
        similar_X = similar_set(dis_data, num_attribs, protected_attribs, dataset.constraint)

        newdata_res = []
        l = len(similar_X)
        for i in range(l):
            newdata_re = new_model.predict(similar_X[i])
            newdata_re = (newdata_re > 0.5).astype(int).flatten()
            newdata_res.append(newdata_re)

        repaired_num = get_repaired_num(newdata_res)
        repair_acc = repaired_num / len(dis_data)
        print("repaire_acc:",m)
        print (repair_acc)