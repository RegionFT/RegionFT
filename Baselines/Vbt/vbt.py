import random
import time
import os
from sklearn.tree import DecisionTreeClassifier
from .utils.vbt_pruning import PruningSampler
from Experiments.common.metrics.readers import save_idi_rows
from .utils.vbt_SearchTree import Tree2SMT
import numpy as np
import logging


class Vbt:
    def __init__(self, black_box_model, protected_list, no_train_data_sample, vbtx_ver="vbt", show_logging=False):
        self.black_box_model = black_box_model
        self.tree2smt = Tree2SMT(feature_names=self.black_box_model.feature_list[:-1], class_name=self.black_box_model.feature_list[-1], protected_att=protected_list, vbtx_ver=vbtx_ver)
        self.train_data = list()
        self.disc_data = list()
        self.test_data = list()
        self.protected_list = [self.black_box_model.feature_list[i] for i in protected_list]
        self.no_train_data_sample = no_train_data_sample
        self.vbtx_ver = vbtx_ver
        self.no_test = 0
        self.no_disc = 0
        self.real_time_consumed = 0
        self.cpu_time_consumed = 0
        if show_logging:
            logging.basicConfig(format="", level=logging.INFO)
        else:
            logging.basicConfig(level=logging.CRITICAL + 1)

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
                                       random_state=None)
        return clf.fit(X, Y)



    def check_disc(self, testdata):
        no_test = len(testdata) // 2
        X = [item[:-1] for item in testdata]
        Y = [int(item[-1]) for item in testdata]
        real_Y = self.black_box_model.predict(X)
        count = 0
        train_data_count = 0
        for i in range(0, no_test * 2, 2):
            equal1 = True if Y[i] == real_Y[i] else False
            equal2 = True if Y[i + 1] == real_Y[i + 1] else False
            #"""
            if not equal1:
                self.train_data.append(X[i] + [real_Y[i]])
                #self.train_data.append(X[i + 1] + [real_Y[i + 1]])
                train_data_count += 1
            if not equal2:
                #self.train_data.append(X[i] + [real_Y[i]])
                self.train_data.append(X[i + 1] + [real_Y[i + 1]])
                train_data_count += 1
            """
            if not (equal1 and equal2):
                self.train_data.append(X[i] + [real_Y[i]])
                self.train_data.append(X[i + 1] + [real_Y[i + 1]])
            """
            if equal1 and equal2:
                self.disc_data.append(testdata[i])
                self.disc_data.append(testdata[i + 1])
                #self.train_data.append(X[i] + [real_Y[i]])
                #self.train_data.append(X[i + 1] + [real_Y[i + 1]])
                self.no_disc += 1
                count += 1
        return train_data_count

    def print_div(self):
        all_div = 0
        loop_t = 1000
        if self.no_test == 0:
            print("0")
            return
        else:
            for _ in range(loop_t):
                r21 = np.array(random.choice(self.test_data))
                r22 = np.array(random.choice(self.test_data))
                all_div += np.count_nonzero(r21-r22)
        print(f"{all_div/loop_t}")

    def test(self, runtime=None, max_leaf_nodes=1000, max_test_data=None, label=("res",0), disc_save_to="DiscData", test_save_to="TestData"):
        start_real_time = time.time()
        start_cpu_time = time.process_time()
        restart_flag = True
        self.no_test = 0
        no_new_train_count = 0
        loop = 0

        logging.info(f"Starting fairness test -- {label[0]}")
        while True:
            loop += 1
            if (runtime is not None) and (time.process_time() - start_cpu_time >= runtime):
                break
            if (max_test_data is not None) and (self.no_test >= max_test_data):
                break

            # Step1: make an approximation model DT of black box model
            if restart_flag:
                self.create_train_data(self.no_train_data_sample)
                restart_flag = False
            DT = self.train_approximate_DT(max_leaf_nodes=max_leaf_nodes)

            # Step2: Construct an SMT formula from DT
            smt_str = self.tree2smt.dt_to_smt(DT)
            param_xor = self.tree2smt.get_parm_xor()

            # Step3: Generate test cases by SMT solver and hashing-based sampling
            if self.vbtx_ver == "vbt":
                sampler = PruningSampler(DT=DT, feature_names=self.black_box_model.feature_list[:-1], smt_str=smt_str, param_xor=param_xor, vbtx_ver="vbt", max_loop=1000, max_path=50, no_of_xor=5, need_only_one_sol=False, need_change_s=True, need_blocking=False, class_list=[self.black_box_model.feature_list[-1]], protected_list=self.protected_list)
            else:
                print("No such version of vbt-x")
                exit()
            satFlag, test_data = sampler.sample()

            if satFlag:
                # if at least one test data is found
                # Step4: Execute test cases against black box model and Step5: Update the training dataset
                self.no_test += len(test_data) // 2
                self.test_data += test_data
                if self.check_disc(test_data) == 0:
                    no_new_train_count += 1
                    if no_new_train_count >= 5:
                        restart_flag = True
                        no_new_train_count = 0
                else:
                    no_new_train_count = 0
            else:
                # if one test data could be found from the decision tree, then restart the loop
                restart_flag = True
                logging.info(f"Restarting due to not finding any test cases in this loop")
            logging.info(f"Loop {loop}: #Disc={len(self.disc_data) // 2}, #Test={self.no_test}")

        self.real_time_consumed = time.time() - start_real_time
        self.cpu_time_consumed = time.process_time() - start_cpu_time
        # save the results of detected discriminatory instances and generated test cases
        logging.info(f"The fairness test is completed")
        if test_save_to is not None and os.path.isdir(test_save_to):
            logging.info(f"Saving the generated test cases to {test_save_to}/{label[0]}-{label[1]}.npy")
            save_idi_rows(f'{test_save_to}/{label[0]}-{label[1]}.npy', self.test_data)
        if disc_save_to is not None and os.path.isdir(disc_save_to):
            logging.info(f"Saving the detected discriminatory instances to {disc_save_to}/{label[0]}-{label[1]}.npy")
            save_idi_rows(f'{disc_save_to}/{label[0]}-{label[1]}.npy', self.disc_data)
        logging.info(f"Finished")
