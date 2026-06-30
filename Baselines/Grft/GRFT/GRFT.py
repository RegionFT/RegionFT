"""
This python file calls functions from experiments.py to reproduce the main experiments of our paper.
"""
import os
import tensorflow as tf
from GeneticAlgorithm import Population
from tensorflow import keras
import numpy as np
import generation_utilities
import time
import random
import itertools
import copy
import argparse
from tensorflow.keras import backend as K

"""
for census income data, age(0), race(6) and gender(7) are protected attributes in 12 features
for german credit data, gender(6) and age(9) are protected attributes in 24 features
for bank marketing data, age(0) is protected attribute in 16 features
"""

def similar_set(X, num_attribs, protected_attribs, constraint):
    # find all similar inputs corresponding to different combinations of protected attributes with non-protected attributes unchanged
    similar_X = []
    protected_domain = []
    for i in protected_attribs:
        protected_domain = protected_domain + [list(range(int(constraint[i][0]), int(constraint[i][1])+1))]
    all_combs = np.array(list(itertools.product(*protected_domain)))
    for i, comb in enumerate(all_combs):
        X_new = copy.deepcopy(X)
        for a, c in zip(protected_attribs, comb):
            X_new[:, a] = c
        similar_X.append(X_new)
    return similar_X

def similar_set_(X, num_attribs, protected_attribs, constraint):
    # find all similar inputs corresponding to different combinations of protected attributes with non-protected attributes unchanged
    similar_X = np.empty(shape=(0, num_attribs))
    X=np.array(X)
    protected_domain = []
    for i in protected_attribs:
        protected_domain = protected_domain + [list(range(int(constraint[i][0]), int(constraint[i][1])+1))]
    all_combs = np.array(list(itertools.product(*protected_domain)))
    for i, comb in enumerate(all_combs):
        X_new = copy.deepcopy(X)
        for a, c in zip(protected_attribs, comb):
            X_new[:, a] = c
        similar_X=np.append(similar_X,X_new,axis=0)
    return similar_X

def local_generation_random(num_attribs, l_num, g_id, protected_attribs, constraint, model, s_l, epsilon, deadline=None):
    # local generation phase of EIDIG
    non_protected_attribs=[]
    for attrib in range(num_attribs):
        if attrib not in protected_attribs:
            non_protected_attribs.append(attrib)
    direction = [-1, 1]
    l_id = np.empty(shape=(0, num_attribs))
    all_gen_l = np.empty(shape=(0, num_attribs))
    try_times = 0
    p=[]
    for i in range(num_attribs):
        if(i not in protected_attribs):
            p.append(i)
    gid=np.array(g_id)
    suc_iter = 0
    if(len(gid)!=0):
        for _ in range(l_num):
            # [RegionFT integration] honor a wall-clock budget so the local
            # phase respects the per-method time limit (only change to upstream
            # GRFT search logic).
            if deadline is not None and time.process_time() >= deadline:
                break
            g0=np.copy(gid)
            try_times += 1
            suc_iter += 1
            a = random.choice(non_protected_attribs)
            s = generation_utilities.random_pick([0.5, 0.5])
            # print(g0)
            g0[:,a] = g0[:,a] + direction[s] * s_l
            g1 = generation_utilities.clip(g0, constraint)
            similar_x1 = similar_set_(g1, num_attribs, protected_attribs, constraint)
            y_pred=(model(tf.constant(similar_x1))>0.5).numpy().astype('int').flatten()
            y_pred=y_pred.reshape(-1,len(g1))
            unique_columns = [len(np.unique(y_pred[:, col])) > 1 for col in range(y_pred.shape[1])]
            index = np.where(unique_columns)[0]
            l_id = np.append(l_id,g1[index],axis=0)
            gid[index] = g0[index]
        l_id = np.array(list(set([tuple(id) for id in l_id])))
    
    return l_id, all_gen_l, try_times


def get_rand_num():
    return random.randint(-5, 5)

def create_image_indvs(seed, num,num_attribs,protected_attribs,constraint):
    non_protected_attribs=[]
    indivs=[]
    indivs.append(seed)
    for attrib in range(num_attribs):
        if attrib not in protected_attribs:
            non_protected_attribs.append(attrib)
    index = np.random.choice(non_protected_attribs,2,replace=False)
    unproattr=len(non_protected_attribs)
    for i in range(num-1):
        temp=seed.copy()
        temp[index[0]]=temp[index[0]]+get_rand_num()
        temp[index[1]]=temp[index[1]]+get_rand_num()
        indivs.append(temp)
    indivs = generation_utilities.clip(indivs, constraint)
    return np.array(indivs)



def untarget_object_func(model, num_attribs, protected_attribs, constraint, target_ratio=0.9):
    def func(indvs, ground_truth):
         # define target
        x_array = np.array(indvs)
        index_result=[]
        fitness = []
        similar_x = similar_set_(x_array,num_attribs,protected_attribs,constraint)
        # print(similar_x)
        # 计算原始输入和变异输入的预测差异
        y_pred = model(tf.constant(x_array)).numpy()
        # y_pred_label = y_pred>0.5).astype("int").flatten()
        pred_similar_x = model(tf.constant(similar_x)).numpy()
        pred_similar_x_label=(pred_similar_x>0.5).astype("int").flatten()
        pred_similar_x_label=pred_similar_x_label.reshape(-1,len(indvs))
        unique_columns = [len(np.unique(pred_similar_x_label[:, col])) > 1 for col in range(pred_similar_x_label.shape[1])]
        new_index_result = np.where(unique_columns)[0]
        y_pred = np.repeat(y_pred, pred_similar_x_label.shape[0], axis=0)
        # print(y_pred.shape)
        # print(pred_similar_x.shape)
        fitness = np.sum(abs(y_pred-pred_similar_x).reshape(-1,len(indvs)), axis=0)
        # print(fitness)
        return x_array[new_index_result], new_index_result, fitness
    return func


def build_mutate_func(num_attribs,protected_attribs,constraint):
    def func(indv):
        non_protected_attribs=[]
        for attrib in range(num_attribs):
            if attrib not in protected_attribs:
                non_protected_attribs.append(attrib)
        index = np.random.choice(non_protected_attribs,4,replace=False)
        mutate_indv=indv.copy()
        mutate_indv[index[0]]=mutate_indv[index[0]]+random.randint(-3,3)
        mutate_indv[index[1]] = mutate_indv[index[1]] + random.randint(-3,3)
        mutate_indv[index[2]] = mutate_indv[index[2]] + random.randint(-3,3)
        mutate_indv = generation_utilities.clip(mutate_indv, constraint)
        return mutate_indv
    return func


def build_mutate_func2(num_attribs,protected_attribs,constraint):
    def func(indv):
        non_protected_attribs=[]
        for attrib in range(num_attribs):
            if attrib not in protected_attribs:
                non_protected_attribs.append(attrib)
        index = np.random.choice(non_protected_attribs,1,replace=False)
        mutate_indv=np.array(indv.copy())
        # print(mutate_indv)
        if random.random() < 0.5:
            mutate_indv[:, index] += 1
        else:
            mutate_indv[:, index] -= 1
        mutate_indv = generation_utilities.clip(mutate_indv, constraint)
        return mutate_indv
    return func

def build_save_func(npy_output, img_output, file_name, seedname, istarget):
    def save_func(indvs, round):

        data, indexes = indvs
        for i, item in enumerate(data):
            name = seedname+ '_'+'_'+ istarget+'_' +'_' + '_'+ str(time.time()) +'_'+ str(round)
            np.save(os.path.join(npy_output, name + '.npy'), data[i])
            x = data[i].astype("int").tolist()
            with open("results_quant.txt", "a+") as file:
                file.write(seedname+";"+str(x)+";"+str(round)+"\n")

    return  save_func



def global_generation(X, seeds, num_attribs, protected_attribs, constraint, model, max_iter, s_g,benchmark,pop_num=100, deadline=None):
    # global generation phase of GRFT
    g_id = np.empty(shape=(0, num_attribs))
    all_gen_g = np.empty(shape=(0, num_attribs))
    try_times = 0
    g_num = len(seeds)
    index = 0
    for i in range(g_num):
        # [RegionFT integration] honor a wall-clock budget so the global phase
        # respects the per-method time limit (only change to upstream GRFT
        # search logic).
        if deadline is not None and time.process_time() >= deadline:
            break
        x1 = seeds[i].copy()
        inds = create_image_indvs(x1, pop_num,num_attribs,protected_attribs,constraint)
        mutation_function = build_mutate_func(num_attribs,protected_attribs, constraint)
        build_mutate_func2_function = build_mutate_func2(num_attribs,protected_attribs, constraint)
        save_function = build_save_func("results", "results", "results_quant.txt", str(i), 'n')
        fitness_compute_function = untarget_object_func(model,num_attribs, protected_attribs, constraint, target_ratio=0)
        pop = Population(inds,mutation_function,fitness_compute_function,save_function,build_mutate_func2_function,num_attribs,first_attack=1,subtotal=10, max_time=1000000000, seed=x1, max_iteration=max_iter)
        try_times=try_times+pop.round
        index += pop.success
        for di in pop.discriminatory:
            g_id = np.append(g_id, [di], axis=0)
    g_id = np.array(list(set([tuple(id) for id in g_id])))
    return g_id, all_gen_g, try_times




def local_generation(num_attribs, l_num, g_id, protected_attribs, constraint, model, s_l, epsilon):
    # local generation phase of GRFT

    direction = [-1, 1]
    l_id = np.empty((0, num_attribs))
    all_gen_l = np.empty((0, num_attribs))
    non_protected_attribs = [attrib for attrib in range(num_attribs) if attrib not in protected_attribs]
    gi_id = g_id.copy()

    for i in range(l_num):
        g0_id = gi_id.copy()
        a = random.choice(non_protected_attribs)
        s = random.choice([0, 1])
        gi_id[:, a] += direction[s] * s_l
        gi_id = generation_utilities.clip(gi_id, constraint)
        
        similar_x = similar_set(gi_id, num_attribs, protected_attribs, constraint)
        y_pred = model(gi_id).numpy().flatten()
        y_pred_label = (y_pred > 0.5).astype(int)
        
        pred_similar_x = [model(sim_x).numpy().flatten() for sim_x in similar_x]
        print(pred_similar_x)
        pred_similar_x_label_0 = (pred_similar_x[0] > 0.5).astype(int)
        
        is_diff_0 = y_pred_label != pred_similar_x_label_0
        index_different = np.nonzero(is_diff_0)[0]
        
        for j in range(1, len(similar_x)):
            pred_similar_x_label = (pred_similar_x[j] > 0.5).astype(int)
            is_diff = y_pred_label != pred_similar_x_label
            different = np.nonzero(is_diff)[0]
            index_different = np.union1d(index_different, different)
        
        l_id = np.append(l_id, gi_id[index_different], axis=0)
        
        mask = np.ones(len(gi_id), dtype=bool)
        mask[index_different] = False
        gi_id[mask] = g0_id[mask]

    l_id = np.array(list({tuple(id) for id in l_id}))
    return l_id, all_gen_l,0


def global_comparison(num_experiment_round, benchmark, X, protected_attribs, constraint, model, decay_list, num_seeds=1000, c_num=4, max_iter=10, s_g=1.0):
    # compare the global phase given the same set of seed
    num_ids = np.array([0] * (len(decay_list) + 1))
    num_iter = np.array([0] * (len(decay_list) + 1))
    time_cost = np.array([0] * (len(decay_list) + 1))
    for i in range(num_experiment_round):
        round_now = i + 1
        print('--- ROUND', round_now, '---')
        num_attribs = len(X[0])
        num_dis = 0
        if num_seeds >= len(X):
            seeds = X
        else:
            clustered_data = generation_utilities.clustering(X, c_num)
            seeds = np.empty(shape=(0, num_attribs))
            for i in range(num_seeds):
                x_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, i%c_num, fashion='Distribution')
                seeds = np.append(seeds, [x_seed], axis=0)
        for seed in seeds:
            similar_seed = generation_utilities.similar_set(seed, num_attribs, protected_attribs, constraint)
            if generation_utilities.is_discriminatory(seed, similar_seed, model):
                num_dis += 1
        print('Given', num_seeds, '(no more than 600 for german credit) seeds,', num_dis, 'of which are individual discriminatory instances.')

        t1 = time.time()
        ids_GRFT, _, total_iter_GRFT = global_generation(X, seeds, num_attribs, protected_attribs, constraint, model, max_iter, s_g)
        t2 = time.time()
        num_ids_GRFT = len(ids_GRFT)
        print('EDS:', 'In', total_iter_GRFT, 'search iterations,', num_ids_GRFT, 'non-duplicate individual discriminatory instances are generated. Time cost:', t2-t1, 's.')
        num_ids[0] += num_ids_GRFT
        num_iter[0] += total_iter_GRFT
        time_cost[0] += t2-t1

    avg_num_ids = num_ids / num_experiment_round
    avg_speed = num_ids / time_cost
    avg_iter = num_iter / num_experiment_round / num_seeds
    print('Results of global phase comparsion on', benchmark, 'given {} seeds'.format(num_seeds), ',averaged on', num_experiment_round, 'rounds:')
    print('EDS:', avg_num_ids[0], 'individual discriminatory instances are generated at a speed of', avg_speed[0], 'per second, and the number of iterations on a singe seed is', avg_iter[0], '.')

    return num_ids, num_iter, time_cost

# num_attribs, l_num, g_id, protected_attribs, constraint, model, update_interval, s_l, epsilon
def individual_discrimination_generation(X, seeds, protected_attribs, constraint, model, l_num,benchmark, max_iter=10, s_g=1.0, s_l=1.0, epsilon=1e-6):
    # complete implementation of GRFT
    # return non-duplicated individual discriminatory instances generated, non-duplicate instances generated and total number of search iterations
    # benchmark=""
    num_attribs = len(X[0])
    t1=time.time()
    g_id, gen_g, g_gen_num = global_generation(X, seeds, num_attribs, protected_attribs, constraint, model, max_iter, s_g,benchmark)
    t2=time.time()
    l_id, gen_l, l_gen_num = local_generation_random(num_attribs, l_num, g_id, protected_attribs, constraint, model, s_l, epsilon)
    if(len(g_id)!=0):
        all_id = np.append(g_id, l_id, axis=0)
    else:
        all_id = g_id
    all_id_nondup = np.array(list(set([tuple(id) for id in all_id])))
    all_gen_nondup = np.empty(shape=(0, num_attribs))
    return all_id_nondup, all_gen_nondup, g_gen_num + l_gen_num,g_id,g_gen_num,t2-t1


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
        ids, _, total_iter,g_id, g_iter, global_time = individual_discrimination_generation(dataset_module.X_train, id_seeds, protected_attribs, constraint, model, l_num,m_path, max_iter=10, s_g=1.0, s_l=1.0, epsilon=1e-6)
        t2 = time.time()
        m_name=m_path.replace("/","_")
        np.save('discriminatory_data/' + m_name+"_"+benchmark + '_ids_GRFT' + '.npy', ids)
        num_ids = len(ids)
        global_num_ids = len(g_id)
        print("GRFT",benchmark,m_path,num_ids,t2-t1,global_num_ids,global_time,ROUND)
        with open(root_dir+filename, 'a+') as save_file:
            print("GRFT",benchmark,m_path,num_ids,t2-t1,global_num_ids,global_time,ROUND,file=save_file)