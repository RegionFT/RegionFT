"""
This python file preprocesses the Census Income Dataset.
"""

import os 
import numpy as np

cache_file= "/tmp/compas_cached_pre1.npz"
no_cache  = not os.path.isfile(cache_file)

if no_cache  :
    
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    
    """
https://github.com/dssg/aequitas
https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
https://github.com/ashryaagr/Fairness.jl/blob/93151f615ced1c9f771cb3151caf6f21ddef3d8f/src/datasets/datasets.jl#LL56-L62
    """
    
    # make outputs stable across runs
    np.random.seed(42)
    tf.random.set_seed(42)
    
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    # load german credit risk dataset
    data_path = os.path.join(cur_dir,'../datasets/compas-scores-two-years.csv')
    # df = pd.read_csv(data_path)
    na_values = []
    df = pd.read_csv(data_path, index_col='id', na_values=na_values)

    def aif360_preprocess(df ):
        '''
        https://aif360.readthedocs.io/en/v0.2.3/_modules/aif360/datasets/compas_dataset.html
        '''
        new_df = df[(df.days_b_screening_arrest <= 30)
                    & (df.days_b_screening_arrest >= -30)
                    & (df.is_recid != -1)
                    & (df.c_charge_degree != 'O')
                    & (df.score_text != 'N/A')]
        return new_df 
    print ("before ai360 filter",len(df))
    
    df = aif360_preprocess(df)
    print ("after ai360 filter",len(df))
    df = df[['sex', 'age', 'age_cat', 'race',
                     'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                     'priors_count', 'c_charge_degree', 
                     #'c_charge_desc',
                     'two_year_recid']]

    del df["age"]

    names= list(df.columns)
    pretected_attr= ["race","sex","age_cat"] #same as age
    pretected_attr_int = [names.index(x) for x in pretected_attr]
    # data = df.values
    # vocab_race = ["African-American", "Asian", "Caucasian", "Hispanic", "Native American","Other"]
    # vocab_gender = ["Female", "Male"]
    # vocab_agecat = ["25 - 45", "Greater than 45", "Less than 25"]
    # vocab_c_charge_sex = ["F","M"]
    # map_str_int=lambda x: {z:y for z,y in  list(zip(x,range(len(x)) )) }
    # df_map  = {"race":map_str_int(vocab_race),
    #            "sex":map_str_int(vocab_gender),
    #             "age_cat":map_str_int(vocab_agecat),
    #             "c_charge_degree":map_str_int(vocab_c_charge_sex),
    #             }
    df_replace_map = {'race': {'African-American': 0, 
              'Asian': 1, 'Caucasian': 2, 
              'Hispanic': 3, 'Native American': 4, 
              'Other': 5}, 
      'sex': {'Female': 0, 'Male': 1}, 
      'age_cat': {'25 - 45': 0, 'Greater than 45': 1, 'Less than 25': 2}, 
      'c_charge_degree': {'F': 0, 'M': 1}}
    
    df = df.replace(df_replace_map) 

    # preprocess the original numerical attributes with binning method
    bins_ ={"juv_fel_count": [0, 1],
            "juv_misd_count":[0,1],
            "juv_other_count":[0,1],
            "priors_count":[0,1,2],}#priors_count 0,1,>=2

    for k,bins in bins_.items():
        df[k] = np.digitize(df[k], bins, right=True)
    
    
    # print (df[:10])
    # data = df.values
    # data = data.astype(np.int32)
    df_X=   df[[n for n in names if n!='two_year_recid'] ]
    df_Y=   df[["two_year_recid"]]

    # for n in df.columns:
    #     print (n,"\t",np.unique(df[n],return_counts=True))
    # print (df_Y[:10])
    # print (df_X[:10])
    # exit()
    X= df_X.values.astype(np.int32)
    y= df_Y.values.astype(np.int32)
    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)
    
    # set constraints for each attribute, 117936000 data points in the input space
    constraint = np.vstack((X.min(axis=0), X.max(axis=0))).T
    
    # for census income data, age(0), race(6) and gender(7) are protected attributes in 12 features
    protected_attribs = pretected_attr_int#[0, 6, 7]
    
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
    
    
    
