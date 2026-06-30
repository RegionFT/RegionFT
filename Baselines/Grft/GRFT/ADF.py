"""
This python file reproduces ADF, the state-of-the-art individual discrimination generation algorithm.
The official implementation can be accessed at https://github.com/pxzhang94/ADF.
"""
import argparse
import os
import sys
sys.path.append("backup/evaluation")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import cluster
import itertools
import time
import generation_utilities
from tensorflow.keras import backend as K

def compute_grad(x, model, loss_func=keras.losses.binary_crossentropy):
    # compute the gradient of loss w.r.t input attributes

    x = tf.constant([x], dtype=tf.float32)
    y_pred = tf.cast(model(x) > 0.5, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = loss_func(y_pred, model(x))
    gradient = tape.gradient(loss, x)
    return gradient[0].numpy()


def global_generation(X, seeds, num_attribs, protected_attribs, constraint, model, max_iter, s_g):
    # global generation phase of ADF

    g_id = np.empty(shape=(0, num_attribs))
    all_gen_g = np.empty(shape=(0, num_attribs))
    try_times = 0
    g_num = len(seeds)
    for i in range(g_num):
        x1 = seeds[i].copy()
        for _ in range(max_iter):
            try_times += 1
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            if generation_utilities.is_discriminatory(x1, similar_x1, model):
                g_id = np.append(g_id, [x1], axis=0)
                break
            x2 = generation_utilities.max_diff(x1, similar_x1, model)
            grad1 = compute_grad(x1, model)
            grad2 = compute_grad(x2, model)
            direction = np.zeros_like(X[0])
            sign_grad1 = np.sign(grad1)
            sign_grad2 = np.sign(grad2)
            for attrib in range(num_attribs):
                if attrib not in protected_attribs and sign_grad1[attrib] == sign_grad2[attrib]:
                    direction[attrib] = sign_grad1[attrib]
            x1 = x1 + s_g * direction
            x1 = generation_utilities.clip(x1, constraint)
            all_gen_g = np.append(all_gen_g, [x1], axis=0)
    g_id = np.array(list(set([tuple(id) for id in g_id])))
    return g_id, all_gen_g, try_times

   
def local_generation(num_attribs, l_num, g_id, protected_attribs, constraint, model, s_l, epsilon):
    # local generation phase of ADF

    direction = [-1, 1]
    l_id = np.empty(shape=(0, num_attribs))
    all_gen_l = np.empty(shape=(0, num_attribs))
    try_times = 0
    for x1 in g_id:
        x0 = x1.copy()
        for _ in range(l_num):
            try_times += 1
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            x2 = generation_utilities.find_pair(x1, similar_x1, model)
            grad1 = compute_grad(x1, model)
            grad2 = compute_grad(x2, model)
            p = generation_utilities.normalization(grad1, grad2, protected_attribs, epsilon)
            a = generation_utilities.random_pick(p)
            s = generation_utilities.random_pick([0.5, 0.5])
            x1[a] = x1[a] + direction[s] * s_l
            x1 = generation_utilities.clip(x1, constraint)
            all_gen_l = np.append(all_gen_l, [x1], axis=0)
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            if generation_utilities.is_discriminatory(x1, similar_x1, model):
                l_id = np.append(l_id, [x1], axis=0)
            else:
                x1 = x0.copy()
    l_id = np.array(list(set([tuple(id) for id in l_id])))
    return l_id, all_gen_l, try_times


def individual_discrimination_generation(X, seeds, protected_attribs, constraint, model, l_num, max_iter=10, s_g=1.0, s_l=1.0, epsilon=1e-6):
    # complete implementation of ADF
    # return non-duplicated individual discriminatory instances generated, non-duplicate instances generated and total number of search iterations

    num_attribs = len(X[0])
    t1=time.time()
    g_id, gen_g, g_gen_num = global_generation(X, seeds, num_attribs, protected_attribs, constraint, model, max_iter, s_g)
    t2=time.time()
    l_id, gen_l, l_gen_num = local_generation(num_attribs, l_num, g_id, protected_attribs, constraint, model, s_l, epsilon)
    all_id = np.append(g_id, l_id, axis=0)
    all_gen = np.append(gen_g, gen_l, axis=0)
    all_id_nondup = np.array(list(set([tuple(id) for id in all_id])))
    all_gen_nondup = np.array(list(set([tuple(gen) for gen in all_gen])))
    return all_id_nondup, all_gen_nondup, g_gen_num + l_gen_num,g_id,g_gen_num,t2-t1


def seedwise_generation(X, seeds, protected_attribs, constraint, model, l_num, max_iter=10, s_g=1.0, s_l=1.0, epsilon=1e-6):
    # perform global generation and local generation successively on each single seed

    num_seeds = len(seeds)
    num_gen = np.array([0] * num_seeds)
    num_ids = np.array([0] * num_seeds)
    num_attribs = len(X[0])
    ids = np.empty(shape=(0, num_attribs))
    all_gen = np.empty(shape=(0, num_attribs))
    direction_l = [-1, 1]
    for index, instance in enumerate(seeds):
        x1 = instance.copy()
        flag = False
        for _ in range(max_iter):
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            if generation_utilities.is_discriminatory(x1, similar_x1, model):
                ids = np.append(ids, [x1], axis=0)
                flag = True
                break
            x2 = generation_utilities.max_diff(x1, similar_x1, model)
            grad1 = compute_grad(x1, model)
            grad2 = compute_grad(x2, model)
            direction_g = np.zeros_like(X[0])
            sign_grad1 = np.sign(grad1)
            sign_grad2 = np.sign(grad2)
            for attrib in range(num_attribs):
                if attrib not in protected_attribs and sign_grad1[attrib] == sign_grad2[attrib]:
                    direction_g[attrib] = sign_grad1[attrib]
            x1 = x1 + s_g * direction_g
            x1 = generation_utilities.clip(x1, constraint)
            all_gen = np.append(all_gen, [x1], axis=0)
        if flag == True:
            x0 = x1.copy()
            for _ in range(l_num):
                similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
                x2 = generation_utilities.find_pair(x1, similar_x1, model)
                grad1 = compute_grad(x1, model)
                grad2 = compute_grad(x2, model)
                p = generation_utilities.normalization(grad1, grad2, protected_attribs, epsilon)
                a = generation_utilities.random_pick(p)
                s = generation_utilities.random_pick([0.5, 0.5])
                x1[a] = x1[a] + direction_l[s] * s_l
                x1 = generation_utilities.clip(x1, constraint)
                all_gen = np.append(all_gen, [x1], axis=0)
                similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
                if generation_utilities.is_discriminatory(x1, similar_x1, model):
                    ids = np.append(ids, [x1], axis=0)
                else:
                    x1 = x0.copy()
        nondup_ids = np.array(list(set([tuple(id) for id in ids])))
        nondup_gen = np.array(list(set([tuple(gen) for gen in all_gen])))
        num_gen[index] = len(nondup_gen)
        num_ids[index] = len(nondup_ids)
    return num_gen, num_ids

def time_record(X, seeds, protected_attribs, constraint, model, l_num, record_step, record_frequency, max_iter=10, s_g=1.0, s_l=1.0, epsilon=1e-6):
    # record time consumption
    
    t1 = time.time()
    num_attribs = len(X[0])
    t = np.array([0.0] * record_frequency)
    direction_l = [-1, 1]
    threshold = record_step
    index = 0
    ids = np.empty(shape=(0, num_attribs))
    num_ids = num_ids_before = 0
    for instance in seeds:
        if num_ids >= record_frequency * record_step:
            break
        x1 = instance.copy()
        flag = False
        for i in range(max_iter+1):
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            if generation_utilities.is_discriminatory(x1, similar_x1, model):
                ids = np.append(ids, [x1], axis=0)
                flag = True
                break
            if i == max_iter:
                break
            x2 = generation_utilities.max_diff(x1, similar_x1, model)
            grad1 = compute_grad(x1, model)
            grad2 = compute_grad(x2, model)
            direction_g = np.zeros_like(X[0])
            sign_grad1 = np.sign(grad1)
            sign_grad2 = np.sign(grad2)
            for attrib in range(num_attribs):
                if attrib not in protected_attribs and sign_grad1[attrib] == sign_grad2[attrib]:
                    direction_g[attrib] = sign_grad1[attrib]
            x1 = x1 + s_g * direction_g
            x1 = generation_utilities.clip(x1, constraint)
            t2 = time.time()
        if flag == True:
            ids = np.array(list(set([tuple(id) for id in ids])))
            num_ids = len(ids)
            if num_ids > num_ids_before:
                num_ids_before = num_ids
                if num_ids == threshold:
                    t[index] = t2 - t1
                    threshold += record_step
                    index += 1
                    if num_ids >= record_frequency * record_step:
                        break
            x0 = x1.copy()
            for _ in range(l_num):
                similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
                x2 = generation_utilities.find_pair(x1, similar_x1, model)
                grad1 = compute_grad(x1, model)
                grad2 = compute_grad(x2, model)
                p = generation_utilities.normalization(grad1, grad2, protected_attribs, epsilon)
                a = generation_utilities.random_pick(p)
                s = generation_utilities.random_pick([0.5, 0.5])
                x1[a] = x1[a] + direction_l[s] * s_l
                x1 = generation_utilities.clip(x1, constraint)
                t2 = time.time()
                similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
                if generation_utilities.is_discriminatory(x1, similar_x1, model):
                    ids = np.append(ids, [x1], axis=0)
                    ids = np.array(list(set([tuple(id) for id in ids])))
                    num_ids = len(ids)
                    if num_ids > num_ids_before:
                        num_ids_before = num_ids
                        if num_ids == threshold:
                            t[index] = t2 - t1
                            threshold += record_step
                            index += 1
                            if num_ids >= record_frequency * record_step:
                                break
                else:
                    x1 = x0.copy()
    return t

def generate_seeds(X, c_num=4, num_seeds = 100, fashion='Distribution'):
    num_attribs = len(X[0])
    clustered_data = generation_utilities.clustering(X, c_num)
    id_seeds = np.empty(shape=(0, num_attribs))
    for i in range(100000000):
        x_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, i % c_num, fashion=fashion)
        id_seeds = np.append(id_seeds, [x_seed], axis=0)
        if len(id_seeds) >= num_seeds:
            break
    return id_seeds

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--benchmark', type=str)
    args = parser.parse_args()
    dataset = args.dataset
    model_path = args.model
    benchmark = args.benchmark
    filename = args.filename
    root_dir="results/"

    results = {}
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
    if(dataset=="pre_census_income"):
        from preprocessing import pre_census_income
        dataset_module=pre_census_income

    elif(dataset=="pre_german_credit"):
        from preprocessing import pre_german_credit
        dataset_module=pre_german_credit

    elif(dataset=="pre_bank_marketing"):
        from preprocessing import pre_bank_marketing
        dataset_module=pre_bank_marketing

    elif(dataset=="pre_compas_scores"):
        from preprocessing import pre_compas_scores
        dataset_module=pre_compas_scores

    elif(dataset=="pre_lsac"):
        from preprocessing import pre_lsac
        dataset_module=pre_lsac
    else:
        print("dataset error")

    # model_path = args.model
    attr_lists = {'C-a':[0],'C-r':[6],'C-g': [7],'C-a&r':[0, 6],'C-a&g':[0, 7],'C-r&g':[6, 7],'G-g':[6],'G-a':[9],'G-g&a':[6, 9],'B-a':[0],'compas-g':[5],'compas-r':[4],'compas-r&g':[4,5],'L-g':[9],'L-r':[10],'L-r&g':[10,9]}

    m = model_path
    model = keras.models.load_model(m, custom_objects={'ScaleLayer': ScaleLayer})
    m_path = m
    protected_attribs = attr_lists[benchmark]
    constraint = dataset_module.constraint
    l_num = 1000
    for ROUND in range(1,11):
        id_seeds = generate_seeds(dataset_module.X_train, num_seeds=1000)
        b = os.path.basename(m) +'_'+ benchmark
        t1 = time.time()
        ids, _, total_iter,g_id, g_iter, global_time = individual_discrimination_generation(dataset_module.X_train, id_seeds, protected_attribs, constraint, model, l_num, max_iter=10, s_g=1.0, s_l=1.0, epsilon=1e-6)
        t2 = time.time()
        m_name=m_path.replace("/","_")
        np.save('discriminatory_data/' + m_name+"_"+benchmark + '_ids_ADF' + '.npy', ids)
        num_ids = len(ids)
        global_num_ids = len(g_id)
        print("ADF",benchmark,m_path,num_ids,t2-t1,global_num_ids,global_time,ROUND)
        with open(root_dir+filename, 'a+') as save_file:
            print("ADF",benchmark,m_path,num_ids,t2-t1,global_num_ids,global_time,ROUND,file=save_file)