from __future__ import division
'''
copy from https://github.com/google-research/google-research/blob/master/group_agnostic_fairness/data_utils/CreateLawSchoolDatasetFiles.ipynb
'''

import pandas as pd
import numpy as np
import json
import os,sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

pd.options.display.float_format = '{:,.2f}'.format
dataset_base_dir = '../law_school/'
dataset_file_name = 'lsac.csv'

file_path = os.path.join(dataset_base_dir,dataset_file_name)
with open(file_path, "r") as file_name:
  temp_df = pd.read_csv(file_name)

# Columns of interest  
df = temp_df[['zfygpa','zgpa','DOB_yr','parttime','gender','race','tier','fam_inc','lsat','ugpa','pass_bar','index6040']].copy()
renameColumns={'gender':'sex',
               'index6040':'weighted_lsat_ugpa',
               'fam_inc':'family_income',
               'tier':'cluster_tier',
               'parttime':'isPartTime'}
target_variable = 'pass_bar'
target_value = 'Passed'

# Renaming columns
df = df.rename(columns = renameColumns)
columns = renameColumns.values()

# NaN in 'pass_bar' refer to dropouts. Considering NaN as failing the bar.
df['pass_bar'] = df['pass_bar'].fillna(value=0.0)
df['pass_bar'] = df.apply(lambda x: 'Passed' if x['pass_bar']==1.0 else 'Failed_or_not_attempted', axis=1).astype('category')

df['zfygpa'] = df['zfygpa'].fillna(value=0.0)
df['zgpa'] = df['zgpa'].fillna(value=0.0)
df['DOB_yr'] = df['DOB_yr'].fillna(value=0.0)
df = df.dropna()

# Binarize target_variable
df['isPartTime'] = df.apply(lambda x: 'Yes' if x['isPartTime']==1.0 else 'No', axis=1).astype('category')

# Process protected-column values
race_dict = {3.0:'Black',7.0:'White'}
sex_dict = {'female':'Female','male':'Male'}
df['race'] = df.apply(lambda x: race_dict[x['race']] if x['race'] in race_dict.keys() else 'Other', axis=1).astype('category')
df['sex'] = df.apply(lambda x: sex_dict[x['sex']] if x['sex'] in sex_dict.keys() else 'Other', axis=1).astype('category')

df.head()

train_df, test_df = train_test_split(df, test_size=0.30, random_state=42)

output_file_path = os.path.join(dataset_base_dir,'train.csv')
with open(output_file_path, mode="w") as output_file:
    train_df.to_csv(output_file,index=False,columns=columns,header=False)
    output_file.close()

output_file_path = os.path.join(dataset_base_dir,'test.csv')
with open(output_file_path, mode="w") as output_file:
    test_df.to_csv(output_file,index=False,columns=columns,header=False)
    output_file.close()

feature_names = [
            "zfygpa",  # numerical feature: standardized 1st year GPA
            "zgpa",  # numerical feature: standardized overall GPA
            "DOB_yr",  # numerical feature: year of birth
            "weighted_lsat_ugpa",  # numerical feature: weighted index using 60% of LSAT and 40% UGPA
            "cluster_tier",  # numerical feature: prestige ranking of cluster
            "family_income",  # numerical feature: family income
            "lsat",  # numerical feature: LSAT score
            "ugpa",  # numerical feature: undegraduate GPA
            "isPartTime",  # categorical feature: is part-time status
            "sex",  # categorical feature: sex
            "race",  # categorical feature: race
            "pass_bar"  # binary target variable: has passed bar
        ]
output_file_path = os.path.join(dataset_base_dir,'train_test_with_columnsname.csv')
with open(output_file_path, mode="w") as output_file:
    df.to_csv(output_file,index=False,columns=feature_names,header=True)
    output_file.close()


# print (np.unique(df["pass_bar"]),"pass_bar")
# print (np.unique(df["sex"]),"sex")
# print (np.unique(df["race"]),"race")
# print ("==="*8)


IPS_example_weights_without_label = {
  0: (len(train_df))/(len(train_df[(train_df.race != 'Black') & (train_df.sex != 'Female')])), # 00: White Male
  1: (len(train_df))/(len(train_df[(train_df.race != 'Black') & (train_df.sex == 'Female')])), # 01: White Female
  2: (len(train_df))/(len(train_df[(train_df.race == 'Black') & (train_df.sex != 'Female')])), # 10: Black Male
  3: (len(train_df))/(len(train_df[(train_df.race == 'Black') & (train_df.sex == 'Female')]))  # 11: Black Female
}
  
output_file_path = os.path.join(dataset_base_dir,'IPS_example_weights_without_label.json')
with open(output_file_path, mode="w") as output_file:
    output_file.write(json.dumps(IPS_example_weights_without_label))
    output_file.close()

print(IPS_example_weights_without_label)

IPS_example_weights_with_label = {
0: (len(train_df))/(len(train_df[(train_df[target_variable] != target_value) & (train_df.race != 'Black') & (train_df.sex != 'Female')])), # 000: Negative White Male
1: (len(train_df))/(len(train_df[(train_df[target_variable] != target_value) & (train_df.race != 'Black') & (train_df.sex == 'Female')])), # 001: Negative White Female
2: (len(train_df))/(len(train_df[(train_df[target_variable] != target_value) & (train_df.race == 'Black') & (train_df.sex != 'Female')])), # 010: Negative Black Male
3: (len(train_df))/(len(train_df[(train_df[target_variable] != target_value) & (train_df.race == 'Black') & (train_df.sex == 'Female')])), # 011: Negative Black Female
4: (len(train_df))/(len(train_df[(train_df[target_variable] == target_value) & (train_df.race != 'Black') & (train_df.sex != 'Female')])), # 100: Positive White Male
5: (len(train_df))/(len(train_df[(train_df[target_variable] == target_value) & (train_df.race != 'Black') & (train_df.sex == 'Female')])), # 101: Positive White Female
6: (len(train_df))/(len(train_df[(train_df[target_variable] == target_value) & (train_df.race == 'Black') & (train_df.sex != 'Female')])), # 110: Positive Black Male
7: (len(train_df))/(len(train_df[(train_df[target_variable] == target_value) & (train_df.race == 'Black') & (train_df.sex == 'Female')])), # 111: Positive Black Female
}
  
output_file_path = os.path.join(dataset_base_dir,'IPS_example_weights_with_label.json')
with open(output_file_path, mode="w") as output_file:
    output_file.write(json.dumps(IPS_example_weights_with_label))
    output_file.close()

print(IPS_example_weights_with_label)

cat_cols = train_df.select_dtypes(include='category').columns
vocab_dict = {}
for col in cat_cols:
  vocab_dict[col] = list(set(train_df[col].cat.categories))
  
output_file_path = os.path.join(dataset_base_dir,'vocabulary.json')
with open(output_file_path, mode="w") as output_file:
    output_file.write(json.dumps(vocab_dict))
    output_file.close()
print(vocab_dict)

temp_dict = train_df.describe().to_dict()
mean_std_dict = {}
for key, value in temp_dict.items():
  mean_std_dict[key] = [value['mean'],value['std']]

output_file_path = os.path.join(dataset_base_dir,'mean_std.json')
with open(output_file_path, mode="w") as output_file:
    output_file.write(json.dumps(mean_std_dict))
    output_file.close()
print(mean_std_dict)
