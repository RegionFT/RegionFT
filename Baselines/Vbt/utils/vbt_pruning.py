import os
import re
import random
import math
from z3 import Solver


class PruningSampler:
    '''
    input: DecSmt.smt2

    Output: return Satflag  0: No CEX is found
                            1: have found at least one CEX and added CEXs to Cand-set.csv
    '''

    def __init__(self, DT, feature_names, smt_str, param_xor, vbtx_ver="improved", no_of_xor=5, p=.5, max_path=100, max_loop=1000, need_only_one_sol=True, need_blocking=True, need_change_s=True, class_list=["Class"], protected_list=["sex"]):
        self.smt_str = smt_str
        self.no_of_xor = no_of_xor
        self.p = p
        self.max_path = max_path
        self.max_loop = max_loop
        self.need_only_one_sol = need_only_one_sol
        self.need_blocking = need_blocking
        self.need_change_s = need_change_s
        self.class_list = class_list
        self.vbtx_ver = vbtx_ver
        self.DT = DT
        self.feature_names = feature_names

        not_equal_list = protected_list + class_list
        self.not_equal_list = list()
        for ch in not_equal_list:
            for index in ['','0','1']:
                self.not_equal_list.append(ch+index)
        self.protected_list = list()
        for ch in protected_list:
            for index in ['0','1']:
                self.protected_list.append(ch+index)
        self.blocking_str = ""
        self.res = dict()
        self.samplers = list()

        self.smt2_content = param_xor["smt2_content"]
        self.new_var_list = param_xor["new_var_list"]
        self.old_var_list = param_xor["old_var_list"]
        self.dict_old_to_new = param_xor["dict_old_to_new"]

    def create_input_file(self, cons, in_loop_1 = True):
        # smt2_content to Input.smt2
        smt_str = ""
        smt_str += self.smt2_content["old"]
        smt_str += self.smt2_content["tree"]
        smt_str += self.smt2_content["fairness"]
        #smt_str += self.smt2_content["new"]


        #for lines in self.smt2_content["xor"]:
        #    smt_str += "%s" % lines
        smt_str += "%s\n" % cons

        for lines in self.smt2_content["blocking_loop1"]:
            smt_str += "%s" % lines
        if not in_loop_1:
            smt_str += "%s" % self.smt2_content["blocking_loop2"]
        smt_str += self.smt2_content["check"]
        self.smt_str = smt_str

    def analysis_z3Output(self, in_loop_1 = True):
        # z3の結果 Output.txtを解析する return sat:True  / unsat:False
        # in_loop_1 = True : unsat-> no result found    sat-> creat res
        #             False: unsat-> have just one res  sat-> have another res
        #"""
        solver = Solver()
        solver.from_string(self.smt_str)
        if "unsat" == str(solver.check()):
            return False  # unsat
        elif in_loop_1:
            model = solver.model()
            for item in model:
                self.res[str(item)] = str(model[item])
        return True  # sat

    def have_sol(self):
        self.create_input_file()
        if self.analysis_z3Output():
            self.blocking_str = ""
            for var in self.new_var_list:
                self.blocking_str += " (= " + var + " " + self.res[var] + ")"
            return True
        else:
            return False

    def have_another_sol(self):
        self.smt2_content["blocking_loop2"] = "(assert (not (and%s)))\n" % self.blocking_str
        self.create_input_file(in_loop_1 = False)
        #os.system(r"z3 Input.smt2 > Output.txt")
        self.smt2_content["blocking_loop2"] = ""
        return self.analysis_z3Output(in_loop_1 = False)

    def generate_simple_ins(self):
        res1 = list()
        res2 = list()
        for ovar in self.old_var_list:
            res1.append(int(self.res[ovar+'0']))
            res2.append(int(self.res[ovar+'1']))
        self.samplers.append(res1)
        self.samplers.append(res2)

    def add_blocking(self):
        # for path
        self.smt2_content["blocking_loop1"].append("(assert (not (and%s)))\n" % self.blocking_str)
    
    def clear_data(self):
        #self.smt2_visitor.clear_data()
        self.res = dict()
        self.blocking_str = ""
        if os.path.exists('Input.smt2'):
            os.remove('Input.smt2')
        if os.path.exists('Output.txt'):
            os.remove('Output.txt')

    def get_Branch_cons(self, sample_data, no):
        cons = list()
        clf = self.DT
        decision_path = clf.decision_path(sample_data)

        node_indicator = decision_path.toarray()[0]

        for i, (node, indicator) in enumerate(zip(clf.tree_.children_left, node_indicator)):
            if indicator == 1:
                index = clf.tree_.feature[node]
                if clf.tree_.feature[node] == -2:
                    break
                else:
                    threshold = clf.tree_.threshold[node]
                    if sample_data[0][index] <= threshold:
                        cons.append(f"(assert (not (<= {self.feature_names[index]}{no} {int(threshold)})))")
                    else:
                        cons.append(f"(assert (not (> {self.feature_names[index]}{no} {int(threshold)})))")
        return cons


    def sample(self):
        # main loop
        satFlag = False
        #solver = Solver()
        #solver.from_string(self.smt_str)
        if not self.analysis_z3Output():
            return False, [] # unsat

        # DataInstance Pruning
        cons_list = list()
        for index in ["0", "1"]:
            for var_name in self.old_var_list:
                if var_name != "Class":
                    var = var_name + index
                    cons_list.append(f"(assert (not (= {var} {self.res[var]})))")

        cons_list = list(dict.fromkeys(cons_list))
        for cons in cons_list:
            self.create_input_file(cons)
            satFlag = self.analysis_z3Output()
            if satFlag:
                self.generate_simple_ins()
                if self.need_blocking:
                    self.add_blocking()

        # Branch Pruning
        cons_list = list()
        for index in ["0", "1"]:
            res_list = list()
            for var_name in self.old_var_list:
                if var_name != "Class":
                    var = var_name + index
                    res_list.append(int(self.res[var]))
            res_list = [res_list]
            cons_list += self.get_Branch_cons(res_list, index)

        cons_list = list(dict.fromkeys(cons_list))
        for cons in cons_list:
            self.create_input_file(cons)
            satFlag = self.analysis_z3Output()
            if satFlag:
                self.generate_simple_ins()
                if self.need_blocking:
                    self.add_blocking()

        self.clear_data()
        if len(self.samplers) == 0:
            return False, []
        else:
            return True, self.samplers
