import numpy as np
import os
import random
import time
import lime
from lime.lime_tabular import LimeTabularExplainer
from .Genetic_Algorithm import GA
import logging
from Experiments.common.metrics.readers import save_idi_rows


def ConstructExplainer(train_vectors, feature_names, class_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(train_vectors, feature_names=feature_names,
                                                       class_names=class_names, discretize_continuous=False)
    return explainer


class Global_Discovery(object):
    def __init__(self, stepsize=1):
        self.stepsize = stepsize

    def __call__(self, iteration, params, input_bounds, sensitive_param):
        s = self.stepsize
        samples = []
        while len(samples) < iteration:
            x = np.zeros(params)
            for i in range(params):
                random.seed(time.time())
                x[i] = random.randint(input_bounds[i][0], input_bounds[i][1])
            x[sensitive_param - 1] = 0
            samples.append(x)
        return samples


class ExpGA:
    def __init__(self, black_box_model, protected_list, protected_name, threshold_l, original_data, show_logging=False):
        self.model = black_box_model
        self.sensitive_param = protected_list[0] + 1
        self.X = original_data

        self.global_disc_inputs = set()
        self.global_disc_inputs_list = []
        self.local_disc_inputs = set()
        self.local_disc_inputs_list = []
        self.tot_inputs = set()
        self.local_inputs = set()
        self.location = np.zeros(21)
        self.threshold_l = threshold_l
        self.threshold = 0

        self.input_bounds = self.model.data_range
        self.feature_names = self.model.feature_list[:-1]
        self.class_names = ['no', 'yes']
        self.sens_name = protected_name
        self.params = len(self.feature_names)

        self.no_disc = 0
        self.no_test = 0
        self.real_time_consumed = 0
        self.cpu_time_consumed = 0
        self.local_real_time_consumed = 0
        self.local_cpu_time_consumed = 0

        if show_logging:
            logging.basicConfig(format="", level=logging.INFO)
        else:
            logging.basicConfig(level=logging.CRITICAL + 1)

    def Searchseed(self, model, feature_names, sens_name, explainer, train_vectors, num, X_ori):
        seed = []
        for x in train_vectors:
            self.tot_inputs.add(tuple(x))
            exp = explainer.explain_instance(x, model.predict_proba, num_features=num)
            explain_labels = exp.available_labels()
            exp_result = exp.as_list(label=explain_labels[0])
            rank = []
            for j in range(len(exp_result)):
                rank.append(exp_result[j][0])
            loc = rank.index(sens_name)
            self.location[loc] = self.location[loc] + 1
            if loc < self.threshold_l:
                seed.append(x)
                imp = []
                for item in feature_names:
                    pos = rank.index(item)
                    imp.append(exp_result[pos][1])
            if len(seed) >= 100:
                return seed
        return seed

    def evaluate_local(self, inp):
        inp0 = [int(i) for i in inp]
        self.tot_inputs.add(tuple(inp0))
        self.local_inputs.add(tuple(inp0))
        pre0 = 0
        pre1 = 0
        for val in range(self.input_bounds[self.sensitive_param - 1][0], self.input_bounds[self.sensitive_param - 1][1] + 1):
            if val != inp[self.sensitive_param - 1]:
                inp1 = [int(i) for i in inp]
                inp1[self.sensitive_param - 1] = val

                inp0 = np.asarray(inp0)
                inp0 = np.reshape(inp0, (1, -1))

                inp1 = np.asarray(inp1)
                inp1 = np.reshape(inp1, (1, -1))

                out0 = self.model.predict(inp0)
                out1 = self.model.predict(inp1)

                #pre0 = self.model.predict_proba(inp0)[0]
                #pre1 = self.model.predict_proba(inp1)[0]

                # print(abs(pre0 - pre1)[0]

                if (abs(out0 - out1) > self.threshold and (tuple(map(tuple, inp0)) not in self.global_disc_inputs)
                        and (tuple(map(tuple, inp0)) not in self.local_disc_inputs)):
                    self.local_disc_inputs.add(tuple(map(tuple, list(inp0))))
                    self.local_disc_inputs_list.append(inp0.tolist()[0] + out0.tolist())
                    self.local_disc_inputs_list.append(inp1.tolist()[0] + out1.tolist())
                    self.no_disc += 1
                    # print(pre0, pre1)
                    # print(out1, out0)

                    # print("Percentage discriminatory inputs - " + str(
                    #     float(len(local_disc_inputs_list)) / float(len(tot_inputs)) * 100))
                    # print("Total Inputs are " + str(len(tot_inputs)))
                    # print("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)))

                    return 2 * abs(out1 - out0) + 1
                    # return abs(pre0-pre1)
        # return abs(pre0-pre1)
        return 2 * abs(out1 - out0) + 1

        # return not abs(out0 - out1) > threshold
        # for binary classification, we have found that the
        # following optimization function gives better results

    def xai_fair_testing(self, max_global=1000, max_local=1000, seed_num=None, runtime=None, label=("res",0), disc_save_to="DiscData", test_save_to="TestData"):
        logging.info(f"Starting fairness test -- {label[0]}")
        start_real_time = time.time()
        start_cpu_time = time.process_time()

        # prepare the testing data and model
        logging.info('Starting Global Search')
        global_discovery = Global_Discovery()

        train_samples = global_discovery(max_global, self.params, self.input_bounds, self.sensitive_param)
        train_samples = np.array(train_samples)
        # train_samples = X[np.random.choice(X.shape[0], max_global, replace=False)]

        np.random.shuffle(train_samples)

        logging.info('Starting Searchseed')
        explainer = ConstructExplainer(self.X, self.feature_names, self.class_names)
        seed = self.Searchseed(self.model, self.feature_names, self.sens_name, explainer, train_samples, self.params, self.X)

        if seed_num is not None:
            # make sure the num of seed is equal to seed_num
            MaxTry = 100
            #for _ in range(MaxTry):
            while True:
                if len(seed) == seed_num:
                    break
                elif len(seed) > seed_num:
                    seed = seed[:seed_num]
                    break
                else:
                    train_samples = global_discovery(max_global, self.params, self.input_bounds, self.sensitive_param)
                    train_samples = np.array(train_samples)
                    np.random.shuffle(train_samples)
                    seed_add = self.Searchseed(self.model, self.feature_names, self.sens_name, explainer, train_samples, self.params, self.X)
                    seed += seed_add

        logging.info('Finish Searchseed')
        for inp in seed:
            inp0 = [int(i) for i in inp]
            inp0 = np.asarray(inp0)
            inp0 = np.reshape(inp0, (1, -1))
            self.global_disc_inputs.add(tuple(map(tuple, inp0)))
            self.global_disc_inputs_list.append(inp0.tolist()[0])

        logging.info("Finished Global Search")
        logging.info('length of total input is:' + str(len(self.tot_inputs)))
        logging.info('length of global discovery is:' + str(len(self.global_disc_inputs_list)))


        logging.info(f"Total time: cpu: {time.process_time() - start_cpu_time}, real: {time.time() - start_real_time}")

        # Local Search
        logging.info("Starting Local Search")
        start_local_real_time = time.time()
        start_local_cpu_time = time.process_time()

        nums = self.global_disc_inputs_list
        DNA_SIZE = len(self.input_bounds)
        cross_rate = 0.9
        mutation = 0.05
        iteration = max_local
        ga = GA(nums=nums, bound=self.input_bounds, func=self.evaluate_local, DNA_SIZE=DNA_SIZE, cross_rate=cross_rate,
                mutation=mutation)  # for random

        loop = 0
        interval = 50
        while True:
            if (iteration is not None) and (loop >= iteration):
                break
            if (runtime is not None) and (time.process_time() - start_cpu_time >= runtime):
                break
            ga.evolution()
            if loop % interval == 0:
                self.no_test = len(self.local_inputs)
                logging.info(f"Loop {loop}: #Disc={self.no_disc}, #Test={self.no_test}")
            loop += 1

        self.no_test = len(self.local_inputs)
        self.real_time_consumed = time.time() - start_real_time
        self.cpu_time_consumed = time.process_time() - start_cpu_time
        self.local_real_time_consumed = time.time() - start_local_real_time
        self.local_cpu_time_consumed = time.process_time() - start_local_cpu_time
        # save the results of detected discriminatory instances and generated test cases
        logging.info(f"The fairness test is completed")
        if test_save_to is not None and os.path.isdir(test_save_to):
            logging.info(f"Saving the generated test cases to {test_save_to}/{label[0]}-{label[1]}.npy")
            save_idi_rows(f'{test_save_to}/{label[0]}-{label[1]}.npy', self.local_inputs)
        if disc_save_to is not None and os.path.isdir(disc_save_to):
            logging.info(f"Saving the detected discriminatory instances to {disc_save_to}/{label[0]}-{label[1]}.npy")
            save_idi_rows(f'{disc_save_to}/{label[0]}-{label[1]}.npy', self.local_disc_inputs_list)
        logging.info(f"Finished")
