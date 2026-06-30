import deeplift
from deeplift.conversion import kerasapi_conversion as kc
import os
import joblib
import numpy as np

def get_relevance(model_path, inputs, no_relu=True, save_path=None):

    if save_path is not None and os.path.exists(save_path):
        return joblib.load(save_path)

    deeplift_model = \
        kc.convert_model_from_saved_files(
            model_path,
            nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)

    scores = []
    for layer_id, layer in enumerate(deeplift_model.get_layers()):
        if no_relu and type(layer) is deeplift.layers.activations.ReLU:
            continue
        deeplift_contribs_func = deeplift_model.get_target_contribs_func(
            find_scores_layer_idx=layer_id,
            target_layer_idx=-2)
        layer_scores = np.array(deeplift_contribs_func(task_idx=0,
                                                 input_data_list=[inputs],
                                                 batch_size=10,
                                                 progress_update=1000))
        scores.append(layer_scores)
    if save_path is not None:
        joblib.dump(scores, save_path)
    return scores

def get_critical_neurons(scores, percent):
    results = []
    layer_num = len(scores)
    for layer_id in range(layer_num):
        layer_gen_socres = scores[layer_id] * -1
        num = max(1, int(layer_gen_socres.shape[-1] * percent))
        sort_gen_indexes = np.argsort(layer_gen_socres)
        results.append( sort_gen_indexes[:, 0:num])
    return results

def get_similarity(scores1, scores2, percent):
    layers = []
    layer_num = len(scores1)
    len_sample = len(scores1[0])

    orders = []
    for layer_id in range(layer_num):
        layer_gen_socres = scores2[layer_id] * -1
        layer_ori_socres = scores1[layer_id] * -1
        sort_ori_indexes = np.argsort(layer_ori_socres)
        sort_gen_indexes = np.argsort(layer_gen_socres)

        orders.append((sort_ori_indexes, sort_gen_indexes))

        num = max(1, int(layer_gen_socres.shape[-1] * percent))

        top_n_ori = sort_ori_indexes[:, 0:num]
        top_n_gen = sort_gen_indexes[:, 0:num]
        jaccard = []
        for i in range(len_sample):
            a = top_n_gen[i]
            b = top_n_ori[i]
            inter = np.intersect1d(a, b)

            jaccard.append(len(inter) / len(a))
        layers.append(round(np.average(jaccard), 4))
    return layers, orders