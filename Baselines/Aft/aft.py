import os
import random
import time
from sklearn.tree import DecisionTreeClassifier
from .utils.PathSearcher import PathSearcher, IntervalPool
from Experiments.common.metrics.readers import save_idi_rows
import logging
import itertools


class AFT:
    def __init__(self, black_box_model, protected_list, no_train_data_sample, show_logging=False):
        self.black_box_model = black_box_model
        self.train_data = list()
        self.disc_data = list()
        self.test_data = list()
        self.protected_list = [self.black_box_model.feature_list[i] for i in protected_list]
        self.protected_list_no = protected_list
        self.no_train_data_sample = no_train_data_sample
        self.no_test = 0
        self.no_disc = 0
        self.real_time_consumed = 0
        self.cpu_time_consumed = 0
        self.protected_value_comb = self.generate_protected_value_combination()
        self.no_prot = len(protected_list)
        if show_logging:
            logging.basicConfig(format="", level=logging.INFO)
        else:
            logging.basicConfig(level=logging.CRITICAL + 1)

    def generate_protected_value_combination(self):
        res = list()
        for index_protected in self.protected_list_no:
            MinMax = self.black_box_model.data_range[index_protected]
            res.append(list(range(MinMax[0], MinMax[1] + 1)))
        return list(itertools.product(*res))

    def create_train_data(self, num):
        self.train_data = list()
        black_model = self.black_box_model
        data_range = black_model.data_range
        for _ in range(num):
            temp = list()
            for i in range(black_model.no_attr):
                temp.append(random.randint(data_range[i][0], data_range[i][1]))
            temp.append(int(black_model.predict([temp])))
            self.train_data.append(temp)

    def train_approximate_DT(self, max_leaf_nodes):
        X = [item[:-1] for item in self.train_data]
        Y = [item[-1] for item in self.train_data]
        clf = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=None, min_samples_split=2,
                                       min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                                       random_state=None, max_leaf_nodes=max_leaf_nodes)
        return clf.fit(X, Y)

    def test(self, runtime=None, max_leaf_nodes=1000, max_test_data=None, label=("res",0),
             dt_search_mode="random+flip", check_type="themis", MaxTry=10000, MaxDiscPathPair=100, max_train_data_each_path=10,
             max_sample_each_path=100, disc_save_to="DiscData", test_save_to="TestData"):
        start_real_time = time.time()
        start_cpu_time = time.process_time()
        restart_flag = True
        self.no_test = 0
        no_new_train_count = 0
        loop = 0
        IntervalP = IntervalPool()

        # Main loop of AFT (Algorithm 2 in paper)
        logging.info(f"Starting fairness test -- {label[0]}")
        while True:
            loop += 1
            if (runtime is not None) and (time.process_time() - start_cpu_time >= runtime):
                break
            if (max_test_data is not None) and (self.no_test >= max_test_data):
                break

            # Generate training data from input space of CuT at random.
            if restart_flag:
                self.create_train_data(self.no_train_data_sample)
                restart_flag = False

            # (Re-)train an approximate decision tree model with training data, as an approximation of CuT.
            DT = self.train_approximate_DT(max_leaf_nodes=max_leaf_nodes)

            # Generate test cases and then identify discriminatory instances through path samping and random search
            # (see the function sampler.sample()).
            sampler = PathSearcher(DT=DT, CuT=self.black_box_model, data_range=self.black_box_model.data_range,
                                   protected_value_comb=self.protected_value_comb, protected_list_no=self.protected_list_no, IntervalP=IntervalP)
            satFlag = sampler.sample(dt_search_mode=dt_search_mode, check_type=check_type, MaxTry=MaxTry, MaxDiscPathPair=MaxDiscPathPair,
                                     max_train_data_each_path=max_train_data_each_path, max_sample_each_path=max_sample_each_path)

            if satFlag:
                # If at least one test case is found,
                # update test cases and discriminatory instances
                test_data = sampler.get_test_data()
                self.no_test += len(test_data) // 2
                self.test_data += test_data
                new_disc_data = sampler.get_disc_data()
                self.no_disc += len(new_disc_data) // 2
                if len(new_disc_data) == 0:
                    # If no one discriminatory instance is found in this iteration, then restart
                    restart_flag = True
                    logging.info(f"Restarting due to not finding any discriminatory data in this loop")
                    logging.info(f"Loop {loop}: #Disc={self.no_disc}, #Test={self.no_test}")
                    continue
                else:
                    self.disc_data += new_disc_data

                # Update the training data
                new_train_data = sampler.get_train_data()
                self.train_data += new_train_data
                if len(new_train_data) == 0:
                    no_new_train_count += 1
                    if no_new_train_count >= 5:
                        restart_flag = True
                        no_new_train_count = 0
                else:
                    no_new_train_count = 0
            else:
                # If no one test case could be found from the decision tree, then restart the loop
                restart_flag = True
                logging.info(f"Restarting due to not finding any test cases in this loop")
            logging.info(f"Loop {loop}: #Disc={self.no_disc}, #Test={self.no_test}")

        self.real_time_consumed = time.time() - start_real_time
        self.cpu_time_consumed = time.process_time() - start_cpu_time
        # Save the results of identified discriminatory instances and generated test cases
        logging.info(f"The fairness test is completed")
        if test_save_to is not None and os.path.isdir(test_save_to):
            logging.info(f"Saving the generated test cases to {test_save_to}/{label[0]}-{label[1]}.npy")
            save_idi_rows(f'{test_save_to}/{label[0]}-{label[1]}.npy', self.test_data)
        if disc_save_to is not None and os.path.isdir(disc_save_to):
            logging.info(f"Saving the detected discriminatory instances to {disc_save_to}/{label[0]}-{label[1]}.npy")
            save_idi_rows(f'{disc_save_to}/{label[0]}-{label[1]}.npy', self.disc_data)
        logging.info(f"Finished")
