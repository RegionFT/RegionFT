"""Data reader for Law School dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
@article{lahoti2020fairness,
  title={Fairness without demographics through adversarially reweighted learning},
  author={Lahoti, Preethi and Beutel, Alex and Chen, Jilin and Lee, Kang and Prost, Flavien and Thain, Nithum and Wang, Xuezhi and Chi, Ed H},
  journal={arXiv preprint arXiv:2006.13114},
  year={2020}
}
'''

#copy from https://github.com/google-research/google-research/tree/master/group_agnostic_fairness


# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3


import os 
import numpy as np

cache_file= "/tmp/lsac_cached.npz"
no_cache  = not os.path.isfile(cache_file)

if no_cache  :

    import numpy as np
    import pandas as pd

    # import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from collections import defaultdict
    from sklearn import preprocessing as sk_preprocessing
    import  random 

    """
    https://github.com/google-research/google-research/tree/master/group_agnostic_fairness/data_utils
    """
    FEATURES_CLASSIFICATION  = [
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
            # "pass_bar"  # binary target variable: has passed bar
        ]
    CONT_VARIABLES = ["zfygpa",  # numerical feature: standardized 1st year GPA
            "zgpa",  # numerical feature: standardized overall GPA
            "DOB_yr",  # numerical feature: year of birth
            "weighted_lsat_ugpa",  # numerical feature: weighted index using 60% of LSAT and 40% UGPA
            "cluster_tier",  # numerical feature: prestige ranking of cluster
            "family_income",  # numerical feature: family income
            "lsat",  # numerical feature: LSAT score
            "ugpa",
            "race",#3 
            ] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "pass_bar" # the decision variable
    SENSITIVE_ATTRS = ["race","sex"]

    # make outputs stable across runs
    np.random.seed(42)
    # tf.random.set_seed(42)
    random.seed(42)

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    # load german credit risk dataset
    data_path = os.path.join(cur_dir,'../datasets/law_school/train_test_with_columnsname.csv')
    # df = pd.read_csv(data_path)
    na_values = []
    df = pd.read_csv(data_path)#, index_col='id', na_values=na_values)
    df = df.dropna(subset=["race","sex","pass_bar"]) # dropping missing vals

    vob_isPartTime={'No':0, 'Yes':1,}
    vob_race={'White':0, 'Other':1,'Black':2, }
    vob_sex={'Female':0, 'Male':1, }
    vob_pass_bar={'Failed_or_not_attempted':0, 'Passed':1}
    
    df_str_2_int = lambda x,vob_dict :vob_dict[x] 
    df["isPartTime"] =df["isPartTime"].apply(df_str_2_int,args=(vob_isPartTime,))
    df["race"] =df["race"].apply(df_str_2_int,args=(vob_race,))
    df["sex"] =df["sex"].apply(df_str_2_int,args=(vob_sex,))
    df["pass_bar"] =df["pass_bar"].apply(df_str_2_int,args=(vob_pass_bar,))
    
    """ Feature normalization and one hot encoding """
    y = df[CLASS_FEATURE].values
    
    del df[CLASS_FEATURE]
    data = df.to_dict('list')

    # convert class label 0 to -1
    # y[y==0] = -1
    X = np.array([]).reshape(len(y), 0) # empty array with num rows same as num examples, will hstack the features to it
    x_control = defaultdict(list)
    # x_control_index = defaultdict(list)


    feature_names = []
    for iidd,attr in enumerate( FEATURES_CLASSIFICATION):
        vals = data[attr]
        if attr in CONT_VARIABLES:
            vals = [float(v) for v in vals]
            if attr!="race":
                vals = sk_preprocessing.scale(vals) # 0 mean and 1 variance  

            vals = np.reshape(vals, (len(y), -1)) # convert from 1-d arr to a 2-d arr with one col

        else: # for binary categorical variables, the label binarizer uses just one var instead of two
            lb = sk_preprocessing.LabelBinarizer()
            lb.fit(vals)
            vals = lb.transform(vals)

        # add to sensitive features dict
        if attr in SENSITIVE_ATTRS:
            x_control[attr] = vals
            # x_control_index[attr] = iidd
        
        if attr == CLASS_FEATURE:
            continue

        # add to learnable features
        X = np.hstack((X, vals))
        feature_names.append(attr)
        # if attr in CONT_VARIABLES: # continuous feature, just append the name
        #     feature_names.append(attr)
        # else: # categorical features
        #     if vals.shape[1] == 1: # binary features that passed through lib binarizer
        #         feature_names.append(attr)
        #     else:
        #         pass 
        #         # for k in lb.classes_: # non-binary categorical features, need to add the names for each cat
        #             # feature_names.append(attr + "_" + str(k))


    # convert the sensitive feature to 1-d array
    x_control = dict(x_control)
    for k in list(x_control.keys()):
        # assert(x_control[k].shape[1] == 1) # make sure that the sensitive feature is binary after one hot encoding
        x_control[k] = np.array(x_control[k]).flatten()

    # sys.exit(1)

    """permute the date randomly"""
    #perm = list(range(0,X.shape[0]))
    #random.shuffle(perm)
    #X = X[perm]
    #y = y[perm]
    #for k in list(x_control.keys()):
    #    x_control[k] = x_control[k][perm]


    # X = add_intercept(X)
    # feature_names = ["intercept"] + feature_names
    assert len(feature_names) == X.shape[1], (feature_names,len(feature_names),"feature_names", X.shape,"X.shape","list.data",list(data),len(list(data)) )
    # print("Features we will be using for classification are:", feature_names, "\n")

    pretected_attr= ["race","sex"]
    pretected_attr_int = [feature_names.index(x) for x in pretected_attr]

    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)
    
    # set constraints for each attribute, 117936000 data points in the input space
    constraint = np.vstack((X.min(axis=0), X.max(axis=0))).T
    
    # for census income data, age(0), race(6) and gender(7) are protected attributes in 12 features
    protected_attribs = pretected_attr_int#[0, 6, 7]

    #double check 
    for  att_id in protected_attribs :
        X[:,att_id]= X[:,att_id].astype(np.int32)
    
    for  att_id in protected_attribs :
        assert len(np.unique(X[:,att_id]) )>1 ,"expect att in [0,1] or [0,1,...N], but get {}".format(np.unique(X[:,att_id]) )
if no_cache:
    np.savez(cache_file,
             X=X,
             y=y,
            X_train=X_train,
            X_val=X_val, y_train=y_train, y_val=y_val,
            X_train_all=X_train_all, 
            X_test=X_test, y_train_all=y_train_all, y_test=y_test,
            constraint=constraint,
            protected_attribs=protected_attribs,
            )    
else:
    data = np.load(cache_file)
    
    X= data["X"]
    y= data["y"]
    
    X_train= data["X_train"]
    X_val= data["X_val"]
    y_train= data["y_train"]
    y_val= data["y_val"]
    X_train_all= data["X_train_all"]
    X_test= data["X_test"]
    y_train_all= data["y_train_all"]
    y_test= data["y_test"]
    constraint= data["constraint"]
    protected_attribs= data["protected_attribs"]
    