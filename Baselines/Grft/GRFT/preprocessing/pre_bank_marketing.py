"""
This python file preprocesses the Census Income Dataset.
"""

import os 
import numpy as np

cache_file= "/tmp/bank_cached.npz"
no_cache  = not os.path.isfile(cache_file)

if no_cache  :
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.model_selection import train_test_split
    
    cur_dir=os.path.dirname(__file__)
        
    """
        https://archive.ics.uci.edu/ml/datasets/bank+marketing
    """
    
    # make outputs stable across runs
    np.random.seed(42)
    tf.random.set_seed(42)
    
    
    def set_table(vocab):
        # set lookup table for categorical attributes
        indices = tf.range(len(vocab), dtype=tf.int64)
        table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
        num_oov_buckets = 1
        table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)
        return table
    
    
    # load bank dataset
    data_path = os.path.join(cur_dir,'datasets/bank-full.csv')
    df = pd.read_csv(data_path, sep=";", encoding='latin-1')
    
    
    # impute the missing values with the most frequent value
    df[df == 'unknown'] = np.nan
    for col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    
    # encode categorical attributes to integers
    data = df.values
    list_index_cat = [1, 2, 3, 4, 6, 7, 8, 10, 15, 16]
    for i in list_index_cat:
        vocab = np.unique(data[:,i])
        table = set_table(vocab)
        data[:, i] = keras.layers.Lambda(lambda cats: table.lookup(cats))(data[:, i])
    data = data.astype(np.int32)
    
    
    # preprocess the original numerical attributes with binning method
    bins_age = [15, 25, 45, 65, 120]
    bins_balance = [-1e4] + [np.percentile(data[:,5], percent, axis=0) for percent in [25, 50, 75]] + [2e5]
    bins_day = [0, 10, 20, 31]
    bins_month = [-1, 2, 5, 8, 11]
    bins_duration = [-1.0] + [np.percentile(data[:,11], percent, axis=0) for percent in [25, 50, 75]] + [6e3]
    bins_campaign = [0.0] + [np.percentile(data[:,12], percent, axis=0) for percent in [25, 50, 75]] + [1e2]
    bins_pdays = [-10.0] + [np.percentile(data[:,13], percent, axis=0) for percent in [25, 50, 75]] + [1e3]
    bins_previous = [-1.0] + [np.percentile(data[:,14], percent, axis=0) for percent in [25, 50, 75]] + [3e2]
    list_index_num = [0, 5, 9, 10, 11, 12, 13, 14]
    list_bins = [bins_age, bins_balance, bins_day, bins_month, bins_duration, bins_campaign, bins_pdays, bins_previous]
    for index, bins in zip(list_index_num, list_bins):
        data[:, index] = np.digitize(data[:, index], bins, right=True)
    
    
    # split data into training data, validation data and test data
    X = data[:, :-1]
    y = data[:, -1]
    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)
    
    
    # set constraints for each attribute, 349920 data points in the input space
    constraint = np.vstack((X.min(axis=0), X.max(axis=0))).T
    
    
    # for bank marketing data, age(0) is the protected attribute in 16 features
    protected_attribs = [0]
    
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
    
    
    