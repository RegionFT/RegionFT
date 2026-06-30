"""
This python file preprocesses the Census Income Dataset.
"""

import os 
import numpy as np

cache_file= "/tmp/adult_cached.npz"
no_cache  = not os.path.isfile(cache_file)

if no_cache  :
    import pandas as pd
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.model_selection import train_test_split
    
    """
        https://www.kaggle.com/vivamoto/us-adult-income-update?select=census.csv
    """

    # make outputs stable across runs
    np.random.seed(42)
    # tf.random.set_random_seed(42)
    
    
    def set_table(vocab):
        # set lookup table for categorical attributes
        indices = tf.range(len(vocab), dtype=tf.int64)
        table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
        num_oov_buckets = 1
        table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)
        return table
    
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    print(__file__)
    data_path = os.path.join(cur_dir, 'datasets/adult.csv' )
    # load adult dataset, and eliminate unneccessary features
    # data_path = ()
    # print (data_path)
    assert os.path.isfile(data_path), data_path 
    
    df = pd.read_csv(data_path, encoding='latin-1')
    df = df.drop(['fnlwgt', 'education'], axis=1)
    
    
    # impute the missing values with the most frequent value
    df[df == '?'] = np.nan
    for col in ['workclass', 'occupation', 'native-country']:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    
    # encode categorical attributes to integers
    data = df.values
    vocab_workclass = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
                        "Local-gov", "State-gov", "Without-pay", "Never-worked"]
    table_workclass = set_table(vocab_workclass)
    vocab_marital_status = ["Married-civ-spouse", "Divorced", "Never-married", "Separated",
                            "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
    table_marital_status = set_table(vocab_marital_status)
    vocab_occupation = ["Tech-support", "Craft-repair", "Other-service", "Sales",
                        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
                        "Transport-moving", "Priv-house-serv", "Protective-serv",
                        "Armed-Forces"]
    table_occupation = set_table(vocab_occupation)
    vocab_relationship = ["Wife", "Own-child", "Husband", "Not-in-family",
                            "Other-relative", "Unmarried"]
    table_relationship = set_table(vocab_relationship)
    vocab_race = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
    table_race = set_table(vocab_race)
    vocab_gender = ["Female", "Male"]
    table_gender = set_table(vocab_gender)
    vocab_native_country = ["United-States", "Cambodia", "England", "Puerto-Rico",
                            "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India",
                            "Japan", "Greece", "South", "China", "Cuba", "Iran",
                            "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
                            "Vietnam", "Mexico", "Portugal", "Ireland", "France",
                            "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti",
                            "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland",
                            "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago",
                            "Peru", "Hong", "Holand-Netherlands"]
    table_native_country = set_table(vocab_native_country)
    vocab_label = ["<=50K", ">50K"]
    table_label = set_table(vocab_label)
    list_index_cat = [1, 3, 4, 5, 6, 7, 11, 12]
    list_table = [table_workclass, table_marital_status, table_occupation,
                    table_relationship, table_race, table_gender,
                    table_native_country, table_label]
    for index, table in zip(list_index_cat, list_table):
        print(data[:, index])
        data[:, index] = keras.layers.Lambda(lambda cats: table.lookup(cats))(data[:, index])
    data = data.astype(np.int32)
    
    
    # preprocess the original numerical attributes with binning method
    bins_age = [15, 25, 45, 65, 120]
    bins_capital_gain = [-1, 0, 99998, 100000]
    bins_capital_loss = [-1, 0, 99998, 100000]
    bins_hours_per_week = [0, 25, 40, 60, 168]
    list_index_num = [0, 8, 9, 10]
    list_bins = [bins_age, bins_capital_gain, bins_capital_loss, bins_hours_per_week]
    for index, bins in zip(list_index_num, list_bins):
        data[:, index] = np.digitize(data[:, index], bins, right=True)
    
    
    # split data into training data, validation data and test data
    X = data[:, :-1]
    y = data[:, -1]
    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)
    
    
    # set constraints for each attribute, 117936000 data points in the input space
    constraint = np.vstack((X.min(axis=0), X.max(axis=0))).T
    
    
    # for census income data, age(0), race(6) and gender(7) are protected attributes in 12 features
    protected_attribs = [0, 6, 7]
    
    
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
    
    
    