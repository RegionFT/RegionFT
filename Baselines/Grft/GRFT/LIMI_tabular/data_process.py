from preprocessing import  pre_census_income
from preprocessing import  pre_german_credit
from preprocessing import  pre_bank_marketing
from preprocessing import  pre_compas_scores
from preprocessing import  pre_lsac
import pandas as pd
import numpy as np
# census_sample = pre_compas_scores.constraint
# print(census_sample)
# print(pre_compas_scores.X[0])
# # print(census_sample[:5])
# # output_file = 'output.csv'
# # np.savetxt(output_file, census_sample, delimiter=',', fmt='%d')
# # output_file= "my_credit_sample.csv"
# # german_sample = pre_german_credit.X_train
# # np.savetxt(output_file, german_sample, delimiter=',', fmt='%d')
# output_file= "my_compas.csv"
# bank_sample = pre_compas_scores.X
# np.savetxt(output_file, bank_sample, delimiter=',',fmt='%f')
# 获取数据集中的列名（header）
# headers = df.columns.tolist()
# print("Headers:", headers)
print(pre_german_credit.headers)
# # print(pre_bank_marketing.protected_attribs)