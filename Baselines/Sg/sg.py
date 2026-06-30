import numpy as np
from sklearn.tree import DecisionTreeClassifier
from queue import PriorityQueue
from z3 import *
import os
import copy
import logging
import joblib
from sklearn.cluster import KMeans
from lime import lime_tabular
import time
from Experiments.common.metrics.readers import save_idi_rows


def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
    else:
        pass


def cluster(cluster_path, original_dataset, cluster_num=4):
    """
    Construct the K-means clustering model to increase the complexity of discrimination
    :param cluster_path: the path of clustering model
    :param original_dataset: the original dataset
    :param cluster_num: the number of clusters to form as well as the number of
            centroids to generate
    :return: the K_means clustering model
    """
    if os.path.exists(cluster_path):
        clf = joblib.load(cluster_path)
    else:
        X = original_dataset
        clf = KMeans(n_clusters=cluster_num, random_state=2019).fit(X)
        joblib.dump(clf, cluster_path)
    return clf


def seed_test_input(cluster_path, original_dataset, cluster_num, limit):
    """
    Select the seed inputs for fairness testing
    :param cluster_path: the path of clustering model
    :param original_dataset: the original dataset
    :param cluster_num: the number of clusters to form as well as the number of
            centroids to generate
    :param limit: the size of seed inputs wanted
    :return: a sequence of seed inputs
    """
    # build the clustering model
    clf = cluster(cluster_path, original_dataset, cluster_num)
    clusters = [np.where(clf.labels_ == i) for i in range(cluster_num)]  # len(clusters[0][0])==32561

    i = 0
    rows = []
    max_size = max([len(c[0]) for c in clusters])
    while i < max_size:
        if len(rows) == limit:
            break
        for c in clusters:
            if i >= len(c[0]):
                continue
            row = c[0][i]
            rows.append(row)
        i += 1
    return np.array(rows)


def getPath(X, cut_model, input, conf):
    """
    Get the path from Local Interpretable Model-agnostic Explanation Tree
    :param X: the whole inputs
    :param cut_model: model to test
    :param input: instance to interpret
    :param conf: the configuration of dataset
    :return: the path for the decision of given instance
    """

    # use the original implementation of LIME
    explainer = lime_tabular.LimeTabularExplainer(X,
                                                  feature_names=conf.feature_name, class_names=conf.class_name, categorical_features=conf.categorical_features,
                                                  discretize_continuous=True)
    o_data, g_data = explainer._LimeTabularExplainer__data_inverse(input, num_samples=5000)
    g_labels = cut_model.predict(g_data)

    # build the interpretable tree
    tree = DecisionTreeClassifier(random_state=2019) #min_samples_split=0.05, min_samples_leaf =0.01
    tree.fit(g_data, g_labels)

    # get the path for decision
    path_index = tree.decision_path(np.array([input])).indices
    path = []
    for i in range(len(path_index)):
        node = path_index[i]
        i = i + 1
        f = tree.tree_.feature[node]
        if f != -2:
            left_count = tree.tree_.n_node_samples[tree.tree_.children_left[node]]
            right_count = tree.tree_.n_node_samples[tree.tree_.children_right[node]]
            left_confidence = 1.0 * left_count / (left_count + right_count)
            right_confidence = 1.0 - left_confidence
            if tree.tree_.children_left[node] == path_index[i]:
                path.append([f, "<=", tree.tree_.threshold[node], left_confidence])
            else:
                path.append([f, ">", tree.tree_.threshold[node], right_confidence])
    return path


def check_for_error_condition(conf, cut_model, t, sens):
    """
    Check whether the test case is an individual discriminatory instance
    :param conf: the configuration of dataset
    :param cut_model: model to test
    :param t: test case
    :param sens: the index of sensitive feature
    :return: whether it is an individual discriminatory instance
    """
    label = cut_model.predict(np.array([t]))[0]
    for val in range(conf.input_bounds[sens-1][0], conf.input_bounds[sens-1][1]+1):
        if val != t[sens-1]:
            tnew = copy.deepcopy(t)
            tnew[sens-1] = val
            label_new = cut_model.predict(np.array([tnew]))[0]
            if label_new != label:
                return True, tnew, label, label_new
    return False, None, None, None


def global_solve(path_constraint, arguments, t, conf):
    """
    Solve the constraint for global generation
    :param path_constraint: the constraint of path
    :param arguments: the name of features in path_constraint
    :param t: test case
    :param conf: the configuration of dataset
    :return: new instance through global generation
    """
    s = Solver()
    for c in path_constraint:
        s.add(arguments[c[0]] >= conf.input_bounds[c[0]][0])
        s.add(arguments[c[0]] <= conf.input_bounds[c[0]][1])
        if c[1] == "<=":
            s.add(arguments[c[0]] <= c[2])
        else:
            s.add(arguments[c[0]] > c[2])

    if s.check() == sat:
        m = s.model()
    else:
        return None

    tnew = copy.deepcopy(t)
    for i in range(len(arguments)):
        if m[arguments[i]] == None:
            continue
        else:
            tnew[i] = int(m[arguments[i]].as_long())
    return tnew.astype('int').tolist()


def local_solve(path_constraint, arguments, t, index, conf):
    """
    Solve the constraint for local generation
    :param path_constraint: the constraint of path
    :param arguments: the name of features in path_constraint
    :param t: test case
    :param index: the index of constraint for local generation
    :param conf: the configuration of dataset
    :return: new instance through global generation
    """
    c = path_constraint[index]
    s = Solver()
    s.add(arguments[c[0]] >= conf.input_bounds[c[0]][0])
    s.add(arguments[c[0]] <= conf.input_bounds[c[0]][1])
    for i in range(len(path_constraint)):
        if path_constraint[i][0] == c[0]:
            if path_constraint[i][1] == "<=":
                s.add(arguments[path_constraint[i][0]] <= path_constraint[i][2])
            else:
                s.add(arguments[path_constraint[i][0]] > path_constraint[i][2])

    if s.check() == sat:
        m = s.model()
    else:
        return None

    tnew = copy.deepcopy(t)
    tnew[c[0]] = int(m[arguments[c[0]]].as_long())
    return tnew.astype('int').tolist()


def average_confidence(path_constraint):
    """
    The average confidence (probability) of path
    :param path_constraint: the constraint of path
    :return: the average confidence
    """
    r = np.mean(np.array(path_constraint)[:,3].astype(float))
    return r


def gen_arguments(conf):
    """
    Generate the argument for all the features
    :param conf: the configuration of dataset
    :return: a sequence of arguments
    """
    arguments = []
    for i in range(conf.params):
        arguments.append(Int(conf.feature_name[i]))
    return arguments

class SG:
    def __init__(self, dataset_name, black_box_model, protected_list, protected_name, original_data, show_logging=False):
        self.model = black_box_model
        self.sensitive_param = protected_list[0] + 1
        self.X = original_data

        self.input_bounds = self.model.data_range
        self.feature_names = self.model.feature_list[:-1]
        self.class_names = ['no', 'yes']
        self.sens_name = protected_name
        self.params = len(self.feature_names)

        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        ensure_directory_exists(f"{current_directory}/clusters/")
        self.cluster_path = f"{current_directory}/clusters/{dataset_name}.pkl"

        self.no_disc = 0
        self.no_test = 0
        self.real_time_consumed = 0
        self.cpu_time_consumed = 0

        if show_logging:
            logging.basicConfig(format="", level=logging.INFO)
        else:
            logging.basicConfig(level=logging.CRITICAL + 1)

    def symbolic_generation(self, limit_test=1000, limit_seed=1000, cluster_num=4, runtime=None, label=("res",0), disc_save_to="DiscData", test_save_to="TestData"):
        """
        The implementation of symbolic generation
        :param cluster_num: the number of clusters to form as well as the number of
                centroids to generate
        :param limit_test: the maximum number of test case
        :param limit_seed: the maximum number of seed data
        """
        logging.info(f"Starting fairness test -- {label[0]}")
        start_real_time = time.time()
        start_cpu_time = time.process_time()

        sensitive_param = self.sensitive_param

        # prepare the data configures
        DataConfig = type('DataConfig', (object,), {})
        data_config = DataConfig()
        data_config.categorical_features = list(range(self.params))
        data_config.class_name = self.class_names
        data_config.feature_name = self.feature_names
        data_config.input_bounds = self.input_bounds
        data_config.params = self.params

        # the rank for priority queue, rank1 is for seed inputs, rank2 for local, rank3 for global
        rank1 = 5
        rank2 = 1
        rank3 = 10
        T1 = 0.3

        # prepare the testing data
        X = self.X
        arguments = gen_arguments(data_config)

        # store the result of fairness testing
        global_disc_inputs = set()
        global_disc_inputs_list = []
        local_disc_inputs = set()
        local_disc_inputs_list = []
        tot_inputs = set()

        # select the seed input for fairness testing
        logging.info('Starting seed input generation')
        inputs = seed_test_input(self.cluster_path, X, cluster_num, limit_seed)
        q = PriorityQueue()  # low push first
        for inp in inputs[::-1]:
            q.put((rank1, X[inp].tolist()))
        logging.info('Finish seed input generation')

        visited_path = []
        l_count = 0
        g_count = 0

        loop = 0
        interval = 50
        while q.qsize() != 0:
            if (runtime is not None) and (time.process_time() - start_cpu_time >= runtime):
                break
            if (limit_test is not None) and (len(tot_inputs) >= limit_test):
                break

            t = q.get()
            t_rank = t[0]
            t = np.array(t[1])
            found, tnew, label_t, label_tnew = check_for_error_condition(data_config, self.model, t, sensitive_param)
            p = getPath(X, self.model, t, data_config)
            temp = copy.deepcopy(t.tolist())
            temp = temp[:sensitive_param - 1] + temp[sensitive_param:]

            tot_inputs.add(tuple(temp))
            if found:
                if (tuple(temp) not in global_disc_inputs) and (tuple(temp) not in local_disc_inputs):
                    if t_rank > 2:
                        global_disc_inputs.add(tuple(temp))
                        global_disc_inputs_list.append(t.tolist() + [label_t])
                        global_disc_inputs_list.append(tnew.tolist() + [label_tnew])
                    else:
                        local_disc_inputs.add(tuple(temp))
                        local_disc_inputs_list.append(t.tolist() + [label_t])
                        local_disc_inputs_list.append(tnew.tolist() + [label_tnew])
                    if (limit_test is not None) and len(tot_inputs) == limit_test:
                        break
                    self.no_disc += 1

                # local search
                for i in range(len(p)):
                    path_constraint = copy.deepcopy(p)
                    c = path_constraint[i]
                    if c[0] == sensitive_param - 1:
                        continue

                    if c[1] == "<=":
                        c[1] = ">"
                        c[3] = 1.0 - c[3]
                    else:
                        c[1] = "<="
                        c[3] = 1.0 - c[3]

                    if path_constraint not in visited_path:
                        visited_path.append(path_constraint)
                        input = local_solve(path_constraint, arguments, t, i, data_config)
                        l_count += 1
                        if input != None:
                            r = average_confidence(path_constraint)
                            q.put((rank2 + r, input))

            # global search
            prefix_pred = []
            for c in p:
                if c[0] == sensitive_param - 1:
                    continue
                if c[3] < T1:
                    break

                n_c = copy.deepcopy(c)
                if n_c[1] == "<=":
                    n_c[1] = ">"
                    n_c[3] = 1.0 - c[3]
                else:
                    n_c[1] = "<="
                    n_c[3] = 1.0 - c[3]
                path_constraint = prefix_pred + [n_c]

                # filter out the path_constraint already solved before
                if path_constraint not in visited_path:
                    visited_path.append(path_constraint)
                    input = global_solve(path_constraint, arguments, t, data_config)
                    g_count += 1
                    if input != None:
                        r = average_confidence(path_constraint)
                        q.put((rank3 - r, input))

                prefix_pred = prefix_pred + [c]

            if loop % interval == 0:
                self.no_test = len(tot_inputs)
                logging.info(f"Loop {loop}: #Disc={self.no_disc}, #Test={self.no_test}")
            loop += 1

        # print the overview information of result
        logging.info("Total Inputs are " + str(len(tot_inputs)))
        logging.info("Total discriminatory inputs of global search- " + str(len(global_disc_inputs)) + str(g_count))
        logging.info("Total discriminatory inputs of local search- " + str(len(local_disc_inputs)) + str(l_count))

        self.no_test = len(tot_inputs)
        self.real_time_consumed = time.time() - start_real_time
        self.cpu_time_consumed = time.process_time() - start_cpu_time
        logging.info(f"The fairness test is completed")
        logging.info(f"Total time: cpu: {self.cpu_time_consumed}, real: {self.real_time_consumed}")

        # save the results of detected discriminatory instances and generated test cases
        if test_save_to is not None and os.path.isdir(test_save_to):
            logging.info(f"Saving the generated test cases to {test_save_to}/{label[0]}-{label[1]}.npy")
            int_tot_inputs = [list(map(int, row)) for row in tot_inputs]
            save_idi_rows(f'{test_save_to}/{label[0]}-{label[1]}.npy', int_tot_inputs)
        if disc_save_to is not None and os.path.isdir(disc_save_to):
            logging.info(f"Saving the detected discriminatory instances to {disc_save_to}/{label[0]}-{label[1]}.npy")
            int_global_disc_inputs_list = [list(map(int, row)) for row in global_disc_inputs_list]
            int_local_disc_inputs_list = [list(map(int, row)) for row in local_disc_inputs_list]
            save_idi_rows(f'{disc_save_to}/{label[0]}-{label[1]}.npy', int_global_disc_inputs_list + int_local_disc_inputs_list)
        logging.info(f"Finished")
