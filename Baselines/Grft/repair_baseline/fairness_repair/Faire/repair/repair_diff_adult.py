import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

from tensorflow import keras
import os
import joblib
import numpy as np
from explain import  get_relevance, get_critical_neurons
import tensorflow as tf
from tensorflow import set_random_seed
from scalelayer import  ScaleLayer
from numpy.random import seed
import itertools
import time
from preprocessing import pre_census_income
import copy

from tensorflow.keras.layers import Lambda
import argparse

seed(1)
set_random_seed(2)

def my_loss_fun(y_true, y_pred):
    # do whatever you want
    return y_pred

def my_loss_fun2(y_true, y_pred):
    # do whatever you want
    return y_pred

def my_slice(x, n, in_len):
    return x[:, n*in_len: (n+1)*in_len]

def construct_model(neurons, top_layer, name, min, max, comb_num, need_weights=True):
    in_len = X_train.shape[1]
    in_shape = X_train.shape[1:][0]
    in_shape2 = X_train.shape[1:][0] * (comb_num)
    
    input = keras.Input(shape=in_shape)
    if need_weights:
        input2 = keras.Input(shape=in_shape2) 
    
    layer1 = keras.layers.Dense(30, name="layer1")
    d1 = ScaleLayer(30, min, max)
    layer2 = keras.layers.Dense(20, name="layer2")
    d2 = ScaleLayer(20, min, max)
    layer3 = keras.layers.Dense(15, name="layer3")
    d3 = ScaleLayer(15, min, max)
    layer4 = keras.layers.Dense(15, name="layer4")
    d4 = ScaleLayer(15, min, max)
    layer5 = keras.layers.Dense(10,name="layer5")
    d5 = ScaleLayer(10, min, max)
    layer6 = keras.layers.Dense(1, activation="sigmoid", name="layer6")

    layer_lst = [layer1, layer2, layer3, layer4, layer5]
    ds = [d1, d2, d3, d4, d5]
    for layer in layer_lst[0: top_layer]:
        layer.trainable = False
        
    x = input
    if need_weights:
        fakes = [None for i in range(comb_num)]
        for i in range(comb_num):
            fakes[i] = Lambda(my_slice, arguments={'n': i, 'in_len': in_len})(input2)
#             fake1 = Lambda(my_slice, arguments={'n': 0, 'in_len': in_len})(input2)
#             fake2 = Lambda(my_slice, arguments={'n': 1, 'in_len': in_len})(input2)
#         fakes = [fake1, fake2]
#             print(fakes[i])
        ori_feature_diff = []
    for i, l in enumerate(layer_lst):
#         print("layer", i)
        x = l(x)
        if need_weights:
            for j in range(comb_num):
#                 print(fakes[j])
                fakes[j] = l(fakes[j])
        if i < top_layer:
            x = ds[i](x)
            if need_weights:
                for j in range(comb_num):
                    fakes[j] = ds[i](fakes[j])
                    ori_feature_diff.append(tf.math.subtract(fakes[j], x))
            
    x = layer6(x)
#     x2 = layer6(x2)
    if not need_weights:
        return keras.Model(input, x)

    w = 0.
    for i, re in enumerate(neurons):
        neg = re[0]
        pos = re[1]
        d = ds[i]
        for m in neg:
#             w = tf.math.add(w, d.weights[0][0][m])
            w = tf.math.add(w, tf.math.abs(d.weights[0][0][m]))
        for n in pos:
            w = tf.math.subtract(w, d.weights[0][0][n])
#             w = tf.math.subtract(w, d.weights[0][0][n])
    new_w = tf.identity(tf.reshape(w, [1, 1]), name=name)
    
    diff = 0.
    for i in range(len(ori_feature_diff)):
        diff = tf.math.add(diff, tf.math.reduce_sum(tf.math.abs(ori_feature_diff[i])))
        
    new_d = tf.identity(tf.reshape(diff, [1, 1]), name=name+"_diff")
        
    model = keras.Model([input, input2], [x, (new_w, new_d)])
    return model

def get_path_dict():
    saved_model_path = "models/finetuned_models_protected_attributes/adult/"
    path_ls = os.listdir(saved_model_path)
    path_dict = {}
    path_dict['r'] = [saved_model_path+p for p in path_ls if "r_adult" in p]
    path_dict['g'] = [saved_model_path+p for p in path_ls if "g_adult" in p]
    path_dict['a'] = [saved_model_path+p for p in path_ls if "a_adult" in p]
    path_dict['r'].sort()
    path_dict['g'].sort()
    path_dict['a'].sort()
    print(path_dict)
    return path_dict

def my_filter(layer_critical, total_num):
    i_unique, i_counts = np.unique(layer_critical, return_counts=True)
    i_rates = i_counts / total_num
    i_sort = np.where(i_rates > args.weight_threshold)[0]  # np.argsort(i_counts*-1)
    i_critical = i_unique[i_sort]
    return i_critical

def similar_set(X, num_attribs, protected_attribs, constraint, name):
    # find all similar inputs corresponding to different combinations of protected attributes with non-protected attributes unchanged
    file_name = f"similar_set/adult_{args.attr}_{name}_similar_set"
    if os.path.exists(file_name):
        similar_X = np.load(file_name)
    else:
        protected_domain = []
        for i in protected_attribs:
            protected_domain = protected_domain + [list(range(constraint[i][0], constraint[i][1]+1))]
        all_combs = np.array(list(itertools.product(*protected_domain)))
        for i, comb in enumerate(all_combs):
            X_new = copy.deepcopy(X)
            for a, c in zip(protected_attribs, comb):
                X_new[:, a] = c
            if i == 0:
                similar_X = copy.deepcopy(X_new)
            else:
                similar_X = np.concatenate((similar_X, X_new), axis=1)        
        np.save(file_name, similar_X)
        print("saved")
    return similar_X, len(all_combs)

def get_repaired_num(newdata_res):
    # identify whether the instance is discriminatory w.r.t. the model
    # print(x.shape)n
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

def get_penalty_awarded(top_n, layer_num, total_num, income_critical, protected_critical_ls):
    neurons = []

    for i in range(layer_num):
        income_layer_critical = income_critical[i].flatten()
        i_critical = my_filter(income_layer_critical, total_num)
        current_penalty = None
        current_awarded = None
        filtered_criticals = []
        for j, a in enumerate(attrs):
            protected_layer_critical = protected_critical_ls[j][i].flatten()
            p_critical = my_filter(protected_layer_critical, total_num)
            filtered_criticals.append(p_critical)
            penalty = np.setdiff1d(p_critical, i_critical)
            awarded = np.setdiff1d(i_critical, p_critical)
            if current_penalty is None:
                current_penalty = penalty
            else:
                current_penalty = np.union1d(current_penalty, penalty)
            if current_awarded is None:
                current_awarded = awarded
            else:
                current_awarded = np.intersect1d(current_awarded, awarded)
            print("current_penalty", current_penalty, "current_awarded", current_awarded)
        neurons.append((current_penalty, current_awarded))
    neurons = neurons[1: (top_n + 1)]
    return neurons

def retrain(k, ps, neurons, para_res):

    name = 'my_name_' + str(top_n) + '_' + str(k)
    new_model = construct_model(neurons, top_n, name, ps[0], ps[1], comb_num)
    new_model.load_weights(args.income_path, by_name=True)

    tf_name = 'tf_op_layer_' + name
    diff_name = tf_name + "_diff"
    losses = {'layer6': 'binary_crossentropy', tf_name: my_loss_fun2, diff_name: my_loss_fun}
    losses_weights = {'layer6': args.l0, tf_name: args.l1, diff_name: args.l2}

    new_model.compile(loss=losses, loss_weights=losses_weights, optimizer="nadam", metrics={'layer6': "accuracy"})
#     print(similar_X_train)
    history = new_model.fit([X_train, similar_X_train], y={'layer6': y_train, tf_name: y_train, diff_name: y_train}, epochs=20,
                            validation_data=([X_val, similar_X_val], {'layer6': y_val, tf_name: y_val, diff_name: y_val}))

    re, _, _ = new_model.predict((X_test, similar_X_test))
    pred_maskmodel = (re > 0.5).astype(int).flatten()

    test_acc = np.sum(pred_maskmodel == y_test) / len(y_test)

    dis_num = 0

    newdata_res = [] 
    for i in range(comb_num):
        similar_X_d = np.zeros_like(dis_data)
        similar_X_d = similar_X_dis[:, (i)*in_len:(i+1)*in_len]
        newdata_re, _, _ = new_model.predict((similar_X_d, similar_X_dis))
        newdata_re = (newdata_re > 0.5).astype(int).flatten()
        newdata_res.append(newdata_re)

    repaired_num = get_repaired_num(newdata_res)
    repair_acc = repaired_num / len(dis_data)
    finals.append((test_acc, repair_acc))
    para_res[ps] = (test_acc, repair_acc)

    if args.saved:
        # model_name = 'models/race_gated_'+str(top_n)+'_'+str(args.percent)+'_'+str(args.weight_threshold)+'.h5'
        model_name = f'models/gated_models_diff/diff_adult_{args.attr}_gated_{str(top_n)}_diff.h5'
        saved_model = construct_model(neurons, top_n, name, ps[0], ps[1], comb_num, need_weights=False)
        saved_model.set_weights(new_model.get_weights())
        saved_model.trainable = True
        tf.keras.models.save_model(saved_model, model_name)

    return para_res

pos_map = { 'a': [0],
            'r': [6],
            'g': [7],
            'a&r': [0, 6],
            'a&g': [0, 7],
            'r&g': [6, 7]
            }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fine-tune models with protected attributes')
    parser.add_argument('--income_path', default='models/adult_model.h5', help='model_path')
    parser.add_argument('--target_model_path', 
                        default='models/retrained_model_EIDIG/adult_EIDIG_INF_retrained_model.h5', help='model_path')
    parser.add_argument('--attr', default='g', help='protected attributes')
    parser.add_argument('--percent', type=float, default=0.3)
    parser.add_argument('--p0', type=float, default=-0.6)
    parser.add_argument('--p1', type=float, default=0.6)
    parser.add_argument('--l0', type=float, default=100)
    parser.add_argument('--l1', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=1)
    parser.add_argument('--weight_threshold', type=float, default=0.2)
    parser.add_argument('--saved', type=bool, default=False)
    parser.add_argument('--adjust_para', type=bool, default=False)
    args = parser.parse_args()
    attrs = args.attr.split("&")

    # data preparations
    path_dict = get_path_dict()
    X_train, X_val, y_train, y_val, constraint = pre_census_income.X_train, \
    pre_census_income.X_val, pre_census_income.y_train, pre_census_income.y_val, pre_census_income.constraint
    X_test, y_test = pre_census_income.X_test, pre_census_income.y_test
    target_model_path = args.target_model_path
    
    in_len = X_train.shape[1]
    data_name = f"discriminatory_data/adult/C-{args.attr}_ids_EIDIG_INF.npy"
    dis_data = np.load(data_name)
    num_attribs = len(X_train[0])
    protected_attribs = pos_map[args.attr]
    similar_X_dis, comb_num = similar_set(dis_data, num_attribs, protected_attribs, constraint, "dis")
    
    similar_X_train, _ = similar_set(X_train, num_attribs, protected_attribs, constraint, "train")
    similar_X_val, _ = similar_set(X_val, num_attribs, protected_attribs, constraint, "val")
    similar_X_test, _ = similar_set(X_test, num_attribs, protected_attribs, constraint, "test")
    income_train_scores = get_relevance(args.income_path, X_train,
                                        save_path=os.path.join('scores/adult', os.path.basename(args.income_path) + ".score"))
    income_critical = get_critical_neurons(income_train_scores, args.percent)
    finals = []
    for top_n in [4]:
        protected_critical_ls = []
        for a in attrs:
            path = path_dict[a][top_n - 1]
            train_scores = get_relevance(path, X_train,  save_path=os.path.join('scores/adult', os.path.basename(path) + ".score"))
            protected_critical = get_critical_neurons(train_scores, args.percent)
            protected_critical_ls.append(protected_critical)

        layer_num = len(income_critical)
        total_num = len(X_train)
        neurons = get_penalty_awarded(top_n, layer_num, total_num, income_critical, protected_critical_ls)

        if args.adjust_para:
            paras = [(a/10, b/10) for a in np.arange(-10, 10, 1) for b in np.arange(a, 10, 1)]
        else:
            paras = [(args.p0, args.p1)]
        print("*"*10, len(paras))
        para_res = dict()
        for k, ps in enumerate(paras):
            retrain(k, ps, neurons, para_res)
        for k in para_res.keys():
            print(k, para_res[k])
            # weights = new_model.get_weights()
    print("Retrain is over!")
#     augmented_model = keras.models.load_model(target_model_path)
#     aug_val = (augmented_model.predict(X_val) > 0.5).astype(int).flatten()
#     aug_data = (augmented_model.predict(dis_data) > 0.5).astype(int).flatten()
#     dis_num = 0
#     newdata_res = []

#     l = len(similar_X)
#     for i in range(l):
#         newdata_re = augmented_model.predict(similar_X[i])
#         newdata_re = (newdata_re > 0.5).astype(int).flatten()
#         newdata_res.append(newdata_re)
#     repaired_num = get_repaired_num(newdata_res)

#     print('Aug', np.sum(aug_val == y_val)/len(y_val))
#     print('Aug', repaired_num/len(dis_data))


#     for k in para_res.keys():
#         if para_res[k][0] > 0.8 and para_res[k][1] > 0.98:
#             print(k, para_res[k])

