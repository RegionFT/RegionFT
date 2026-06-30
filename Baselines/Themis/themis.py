import logging
import os
import random
import time
import copy
import itertools
from Experiments.common.metrics.readers import save_idi_rows


class Themis:
    def __init__(self, black_box_model, protected_list, show_logging=False):
        self.black_box_model = black_box_model
        self.data_range = self.black_box_model.data_range
        self.protected_list = [self.black_box_model.feature_list[i] for i in protected_list]
        self.protected_list_no = protected_list
        self.no_prot = len(protected_list)
        self.protected_value_comb = self.generate_protected_value_combination()
        self.disc_data = list()
        self.test_data = list()

        self.no_test = 0
        self.no_disc = 0
        self.real_time_consumed = 0
        self.cpu_time_consumed = 0
        if show_logging:
            logging.basicConfig(format="", level=logging.INFO)
        else:
            logging.basicConfig(level=logging.CRITICAL + 1)

    def generate_protected_value_combination(self):
        res = list()
        for index_protected in self.protected_list_no:
            MinMax = self.data_range[index_protected]
            res.append(list(range(MinMax[0], MinMax[1] + 1)))
        return list(itertools.product(*res))

    def check_disc(self, test):
        y = int(self.black_box_model.predict([test]))
        self.no_test += 1
        self.test_data.append(test)

        test2 = copy.deepcopy(test)
        comb_to_be_removed = tuple(test[i] for i in self.protected_list_no)
        comb_removed_same = [item for item in self.protected_value_comb if item != comb_to_be_removed]
        random.shuffle(comb_removed_same)
        for combination in comb_removed_same:
            for i in range(self.no_prot):
                test2[self.protected_list_no[i]] = combination[i]

            y2 = int(self.black_box_model.predict([test2]))
            if y != y2:
                self.disc_data.append(test+[y])
                self.disc_data.append(test2+[y2])
                self.no_disc += 1
                break

    def test(self, runtime=None, max_test=10000, max_disc=1000, label=("res",0), disc_save_to="DiscData", test_save_to="TestData"):
        data_range = self.data_range
        logging.info(f"Starting fairness test -- {label[0]}")

        start_real_time = time.time()
        start_cpu_time = time.process_time()
        loop = 0
        interval = 1000
        while True:
            if (runtime is not None) and (time.process_time() - start_cpu_time >= runtime):
                break
            if (max_test is not None) and (loop > max_test):
                break
            if (max_disc is not None) and (self.no_disc >= max_disc):
                break
            loop += 1

            test = list()
            for i in range(self.black_box_model.no_attr):
                test.append(random.randint(data_range[i][0], data_range[i][1]))
            self.check_disc(test)

            if loop % interval == 0:
                logging.info(
                    f"Loop {loop}: #Disc={self.no_disc}, #Test={self.no_test}, Prec={self.no_disc / self.no_test}")

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
