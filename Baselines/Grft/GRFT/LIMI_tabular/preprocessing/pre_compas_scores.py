"""
This python file preprocesses the Census Income Dataset.
"""

import os 
import numpy as np

cache_file= "/tmp/compas_cached.npz"
no_cache  = not os.path.isfile(cache_file)

if no_cache  :
    
    import numpy as np
    import pandas as pd

    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from collections import defaultdict
    from sklearn import preprocessing as sk_preprocessing
    import  random 

    def add_intercept(x):
    
        """ Add intercept to the data before linear classification """
        m,n = x.shape
        intercept = np.ones(m).reshape(m, 1) # the constant b
        return np.concatenate((intercept, x), axis = 1)

    """
https://github.com/dssg/aequitas
https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
https://github.com/ashryaagr/Fairness.jl/blob/93151f615ced1c9f771cb3151caf6f21ddef3d8f/src/datasets/datasets.jl#LL56-L62
    """
    FEATURES_CLASSIFICATION = ["age_cat", "race", "sex", "priors_count", "c_charge_degree"] #features to be used for classification
    CONT_VARIABLES = ["priors_count"] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "two_year_recid" # the decision variable
    SENSITIVE_ATTRS = ["race","sex"]

    # make outputs stable across runs
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    # load german credit risk dataset
    data_path = os.path.join(cur_dir,'../datasets/compas-scores-two-years.csv')
    # df = pd.read_csv(data_path)
    na_values = []
    df = pd.read_csv(data_path, index_col='id', na_values=na_values)
    df = df.dropna(subset=["days_b_screening_arrest"]) # dropping missing vals

    data = df.to_dict('list')
    for k in list(data.keys()):
        data[k] = np.array(data[k])


    keys_list=  list(data.keys())
    print ("keys_list",keys_list)
    """ Filtering the data """

    # These filters are the same as propublica (refer to https://github.com/propublica/compas-analysis)
    # If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested, we assume that because of data quality reasons, that we do not have the right offense. 
    idx = np.logical_and(data["days_b_screening_arrest"]<=30, data["days_b_screening_arrest"]>=-30)


    # We coded the recidivist flag -- is_recid -- to be -1 if we could not find a compas case at all.
    idx = np.logical_and(idx, data["is_recid"] != -1)

    # In a similar vein, ordinary traffic offenses -- those with a c_charge_degree of 'O' -- will not result in Jail time are removed (only two of them).
    idx = np.logical_and(idx, data["c_charge_degree"] != "O") # F: felony, M: misconduct

    # We filtered the underlying data from Broward county to include only those rows representing people who had either recidivated in two years, or had at least two years outside of a correctional facility.
    idx = np.logical_and(idx, data["score_text"] != "NA")

    # we will only consider blacks and whites for this analysis
    idx = np.logical_and(idx, np.logical_or(data["race"] == "African-American", data["race"] == "Caucasian"))

    # select the examples that satisfy this criteria
    for k in list(data.keys()):
        data[k] = data[k][idx]

    """ Feature normalization and one hot encoding """

    # convert class label 0 to -1
    print (df["race"][:10],"race")
    print (df["sex"][:10],"sex")
    print (df["c_charge_degree"][:10],"c_charge_degree")
    y = data[CLASS_FEATURE]
    # y[y==0] = -1

    X = np.array([]).reshape(len(y), 0) # empty array with num rows same as num examples, will hstack the features to it
    x_control = defaultdict(list)
    x_control_index = defaultdict(list)

    feature_names = []
    for iidd,attr in enumerate( FEATURES_CLASSIFICATION):
        vals = data[attr]
        if attr in CONT_VARIABLES:
            vals = [float(v) for v in vals]
            vals = sk_preprocessing.scale(vals) # 0 mean and 1 variance  
            vals = np.reshape(vals, (len(y), -1)) # convert from 1-d arr to a 2-d arr with one col

        else: # for binary categorical variables, the label binarizer uses just one var instead of two
            lb = sk_preprocessing.LabelBinarizer()
            lb.fit(vals)
            vals = lb.transform(vals)

        # add to sensitive features dict
        if attr in SENSITIVE_ATTRS:
            x_control[attr] = vals
            x_control_index[attr] = iidd


        # add to learnable features
        X = np.hstack((X, vals))

        if attr in CONT_VARIABLES: # continuous feature, just append the name
            feature_names.append(attr)
        else: # categorical features
            if vals.shape[1] == 1: # binary features that passed through lib binarizer
                feature_names.append(attr)
            else:
                for k in lb.classes_: # non-binary categorical features, need to add the names for each cat
                    feature_names.append(attr + "_" + str(k))


    # convert the sensitive feature to 1-d array
    x_control = dict(x_control)
    for k in list(x_control.keys()):
        assert(x_control[k].shape[1] == 1) # make sure that the sensitive feature is binary after one hot encoding
        x_control[k] = np.array(x_control[k]).flatten()

    # sys.exit(1)

    """permute the date randomly"""
    #perm = list(range(0,X.shape[0]))
    #random.shuffle(perm)
    #X = X[perm]
    #y = y[perm]
    #for k in list(x_control.keys()):
    #    x_control[k] = x_control[k][perm]


    X = add_intercept(X)

    feature_names = ["intercept"] + feature_names
    assert(len(feature_names) == X.shape[1])
    print("Features we will be using for classification are:", feature_names, "\n")


    pretected_attr= ["race","sex"]
    pretected_attr_int = [feature_names.index(x) for x in pretected_attr]
    
    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)
    
    # set constraints for each attribute, 117936000 data points in the input space
    constraint = np.vstack((X.min(axis=0), X.max(axis=0))).T
    
    # for census income data, age(0), race(6) and gender(7) are protected attributes in 12 features
    protected_attribs = pretected_attr_int#[4,5]
    print(protected_attribs)
if no_cache:
        np.savez(cache_file,
                 X=X,
                 y=y,
                X_train=X_train,
                X_val=X_test, 
                y_train=y_train, 
                y_val=y_test,
                X_train_all=X, 
                y_train_all=y, 
                X_test=X_test, 
                y_test=y_test,
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
    
    
    
