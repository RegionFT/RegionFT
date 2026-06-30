import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

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
from preprocessing import pre_lsac
import tensorflow.keras.backend as KTF
import argparse
from tensorflow.keras import activations


seed(1)
tf.random.set_random_seed(2)
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True 
sess = tf.Session(config=config)

KTF.set_session(sess)

def my_loss_fun(y_true, y_pred):
    # do whatever you want
    return y_pred

def construct_model(neurons, top_layer, name, min, max, need_weights=True):
    in_shape = X_train.shape[1:]
    input = keras.Input(shape=in_shape)
    # layer1 = keras.layers.Dense(30, activation="relu", name="layer1")
    # d1 = ScaleLayer(30, min, max)
    # layer2 = keras.layers.Dense(20, activation="relu", name="layer2")
    # d2 = ScaleLayer(20, min, max)
    # layer3 = keras.layers.Dense(15, activation="relu", name="layer3")
    # d3 = ScaleLayer(15, min, max)
    # layer4 = keras.layers.Dense(15, activation="relu", name="layer4")
    # d4 = ScaleLayer(15, min, max)
    # layer5 = keras.layers.Dense(10, activation="relu", name="layer5")
    # d5 = ScaleLayer(10, min, max)
    # layer6 = keras.layers.Dense(1, activation="sigmoid", name="layer6")
    layer1 = keras.layers.Dense(50, name="layer1")
    d1 = ScaleLayer(50, min, max)
    layer2 = keras.layers.Dense(30, name="layer2")
    d2 = ScaleLayer(30, min, max)
    layer3 = keras.layers.Dense(15, name="layer3")
    d3 = ScaleLayer(15, min, max)
    layer4 = keras.layers.Dense(10, name="layer4")
    d4 = ScaleLayer(10, min, max)
    layer5 = keras.layers.Dense(5,name="layer5")
    d5 = ScaleLayer(5, min, max)
    layer6 = keras.layers.Dense(1, activation="sigmoid", name="layer6")
    
    act = keras.layers.Activation(activations.relu) 
    layer_lst = [layer1, layer2, layer3, layer4, layer5]
    ds = [d1, d2, d3, d4, d5]
    for layer in layer_lst[0: top_layer]:
        layer.trainable = False
        
    x = input
    for i, l in enumerate(layer_lst):
        x = l(x)
        if i < top_layer:
            
            x = ds[i](x)
            x = act(x)
            
    x = layer6(x)

    if not need_weights:
        return keras.Model(input, x)

    w = 0.
    
    for i, re in enumerate(neurons):
        neg = re[0]
        pos = re[1]
        d = ds[i]
    
#         normal = [j for j in range(d.weights[0].shape[1]) if j not in neg and j not in pos]
        normal = [j for j in range(d.weights[0].shape[1])]
        for m in neg:
#             w = tf.math.add(w, tf.math.abs(d.weights[0][0][m]))
            w = tf.math.add(w, d.weights[0][0][m])
#             w = tf.math.subtract(w, d.weights[0][0][m])
#             w = tf.math.add(w, tf.math.sum(tf.math.abs(d.weights[0][0])))
        for n in pos:
            w = tf.math.subtract(w, d.weights[0][0][n])
#             w = tf.math.add(w, tf.math.abs(d.weights[0][0][n]))
#             w = tf.math.add(w, d.weights[0][0][n])
#     for m in range(ds[-1].weights[0].shape[1]):
#         w = tf.math.add(w, tf.math.abs(ds[-1].weights[0][0][m]))
    new_w = tf.identity(tf.reshape(w, [1, 1]), name=name)

    model = keras.Model(input, [x, new_w])
    return model

def get_path_dict():
    saved_model_path = "models/finetuned_models_protected_attributes/lsac/"
    path_ls = os.listdir(saved_model_path)
    path_dict = {}
    path_dict['r'] = [saved_model_path+p for p in path_ls if "r_lsac" in p]
    path_dict['g'] = [saved_model_path+p for p in path_ls if "g_lsac" in p]
    path_dict['r'].sort()
    path_dict['g'].sort()
    print(path_dict)
    return path_dict

def my_filter(layer_critical, total_num):
    i_unique, i_counts = np.unique(layer_critical, return_counts=True)
    i_rates = i_counts / total_num
    i_sort = np.where(i_rates > args.weight_threshold)[0]  # np.argsort(i_counts*-1)
    i_critical = i_unique[i_sort]
    return i_critical

def similar_set(X, num_attribs, protected_attribs, constraint):
    # find all similar inputs corresponding to different combinations of protected attributes with non-protected attributes unchanged
    similar_X = []
    protected_domain = []
    for i in protected_attribs:
        protected_domain = protected_domain + [list(range(int(constraint[i][0]), int(constraint[i][1]+1)))]
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
#             penalty = np.intersect1d(p_critical, i_critical)
            awarded = np.setdiff1d(i_critical, p_critical)
            if current_penalty is None:
                current_penalty = penalty
            else:
                current_penalty = np.union1d(current_penalty, penalty)
            if current_awarded is None:
                current_awarded = awarded
            else:
#                 current_awarded = np.intersect1d(current_awarded, awarded)
                current_awarded = np.union1d(current_awarded, awarded)
        print("current_penalty", current_penalty, "current_awarded", current_awarded)
        neurons.append((current_penalty, current_awarded))
    neurons = neurons[1: (top_n + 1)]
    return neurons

def retrain(k, ps, neurons, para_res):

    name = 'my_name_' + str(top_n) + '_' + str(k)
    new_model = construct_model(neurons, top_n, name, ps[0], ps[1])
    new_model.load_weights(args.income_path, by_name=True)

    tf_name = 'tf_op_layer_' + name
    losses = {'layer6': 'binary_crossentropy', tf_name: my_loss_fun}
    losses_weights = {'layer6': 1.0, tf_name: 1.0}

    new_model.compile(loss=losses, loss_weights=losses_weights, optimizer="nadam", metrics={'layer6': "accuracy"})
    history = new_model.fit(x=X_train, y={'layer6': y_train, tf_name: y_train}, epochs=10,
                            validation_data=(X_val, {'layer6': y_val, tf_name: y_val}))

    re, _ = new_model.predict(X_test)
    pred_maskmodel = (re > 0.5).astype(int).flatten()

    test_acc = np.sum(pred_maskmodel == y_test) / len(y_test)

    # data_re, _ = new_model.predict(data)
    # data_re = (data_re > 0.5).astype(int).flatten()

    if test_acc > args.acc_lb:
        newdata_res = []
        l = len(similar_X)
        for i in range(l):
            newdata_re, _ = new_model.predict(similar_X[i])
            newdata_re = (newdata_re > 0.5).astype(int).flatten()
            newdata_res.append(newdata_re)

        repaired_num = get_repaired_num(newdata_res)
        repair_acc = repaired_num / len(dis_data)
    else:
        repair_acc = 0

    # data_re, _ = new_model.predict(data)
    # data_re = (data_re > 0.5).astype(int).flatten()
    finals.append((test_acc, repair_acc))
    para_res[ps] = (test_acc, repair_acc)

    if args.saved:
        # model_name = 'models/race_gated_'+str(top_n)+'_'+str(args.percent)+'_'+str(args.weight_threshold)+'.h5'
        model_name = f'models/gated_models/lsac_{args.attr}_gated_{str(top_n)}_{str(args.percent)}_{args.weight_threshold}_p{ps[0]}_p{ps[1]}.h5'
        saved_model = construct_model(neurons, top_n, name, ps[0], ps[1], need_weights=False)
        saved_model.set_weights(new_model.get_weights())
        saved_model.trainable = True
        tf.keras.models.save_model(saved_model, model_name)

    return para_res

pos_map = { 
            'r': [10],
            'g': [9],
            'g&r': [9, 10]
            }
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fine-tune models with protected attributes')
    parser.add_argument('--income_path', default='models/lsac_model.h5', help='model_path')
    parser.add_argument('--target_model_path', default='models/retrained_model_EIDIG/lsac_EIDIG_INF_retrained_model.h5', help='model_path')
    parser.add_argument('--attr', default='g', help='protected attributes')
    parser.add_argument('--percent', type=float, default=0.3)
    parser.add_argument('--p0', type=float, default=1)
    parser.add_argument('--p1', type=float, default=1)
    parser.add_argument('--weight_threshold', type=float, default=0.2)
    parser.add_argument('--saved', type=bool, default=False)
    parser.add_argument('--adjust_para', type=bool, default=False)
    parser.add_argument('--acc_lb', type=float, default=0.80)
    args = parser.parse_args()
    attrs = args.attr.split("&")

    
    # data preparations
    path_dict = get_path_dict()
    X_train, X_val, y_train, y_val, constraint = pre_lsac.X_train, \
    pre_lsac.X_val, pre_lsac.y_train, pre_lsac.y_val, pre_lsac.constraint
    
    X_test, y_test = pre_lsac.X_test, pre_lsac.y_test
    target_model_path = args.target_model_path
    data_name = f"discriminatory_data/lsac/lsac-{args.attr}_ids_EIDIG_INF_1.npy"
    if args.attr == "g&r":
        data_name = f"discriminatory_data/lsac/lsac-{args.attr}_ids_EIDIG_5_1.npy"
    dis_data = np.load(data_name)
    num_attribs = len(X_train[0])
    protected_attribs = pos_map[args.attr]
    similar_X = similar_set(dis_data, num_attribs, protected_attribs, constraint)

    income_train_scores = get_relevance(args.income_path, X_train,
                                        save_path=os.path.join('scores/lsac', os.path.basename(args.income_path) + ".score"))
    income_critical = get_critical_neurons(income_train_scores, args.percent)
    finals = []
    for top_n in [4]:
        protected_critical_ls = []
        for a in attrs:
            path = path_dict[a][top_n - 1]
            # path = "models/lsac_race_model_4_0.87.h5"
            train_scores = get_relevance(path, X_train,  save_path=os.path.join('scores/lsac', os.path.basename(path) + ".score"))
            protected_critical = get_critical_neurons(train_scores, args.percent)
            protected_critical_ls.append(protected_critical)

        layer_num = len(income_critical)
        total_num = len(X_train)
        neurons = get_penalty_awarded(top_n, layer_num, total_num, income_critical, protected_critical_ls)
        # paras = [(-1, 1), (-0.8, 1), (-0.5, 1), (-0.2, 1), (0, 1), (-1, 1.5), (-0.8, 1.5), (-0.4, 1.5), (0, 1.5), (-1, 2), (-0.8, 2), (-0.4, 2), (0,2)]
        # paras = [(0.2, 1), (0.5, 1), (0.7,1), (0.9,1)]
        # paras = [(-1, 1)]
        s = time.time()
        if args.adjust_para:
            paras = [(a/10, b/10) for a in np.arange(-11, 0, 1) for b in np.arange(1, 10, 1)]
        else:
            paras = [(-args.p0/20, args.p1/20)]
        print("*"*100, paras)
        para_res = dict()
        for k, ps in enumerate(paras):
            retrain(k, ps, neurons, para_res)
            
        e = time.time()
        print("time", e-s)
        for k in para_res.keys():
            print(k, para_res[k])
            # weights = new_model.get_weights()
            file_path = f'records/lsac_repair_relu/{args.attr}_{args.percent}_{args.weight_threshold}/'
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            file_name = file_path + f'{round(para_res[k][0], 4)}_{round(para_res[k][1], 4)}_{k}.txt'
            with open(file_name, 'w') as f:
                f.write("done")
                
    augmented_model = keras.models.load_model(target_model_path)
    aug_val = (augmented_model.predict(X_val) > 0.5).astype(int).flatten()
    aug_data = (augmented_model.predict(dis_data) > 0.5).astype(int).flatten()
    dis_num = 0
    newdata_res = []

    l = len(similar_X)
    for i in range(l):
        newdata_re = augmented_model.predict(similar_X[i])
        newdata_re = (newdata_re > 0.5).astype(int).flatten()
        newdata_res.append(newdata_re)
    repaired_num = get_repaired_num(newdata_res)

    print('Aug', np.sum(aug_val == y_val)/len(y_val))
    print('Aug', repaired_num/len(dis_data))

    
#     for k in para_res.keys():
#         if para_res[k][0] > 0.8 and para_res[k][1] > 0.98:
#             print(k, para_res[k])

