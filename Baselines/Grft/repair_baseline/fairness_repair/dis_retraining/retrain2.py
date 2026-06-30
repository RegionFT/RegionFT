"""
This python file retrains the original models with augmented training set.
"""


import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import joblib
import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from preprocessing import pre_census_income
from preprocessing import pre_german_credit
from preprocessing import pre_bank_marketing
import train_census_income
import train_german_credit
import train_bank_marketing

def retraining(dataset_name, approach_name, ids):
    # randomly sample 5% of individual discriminatory instances generated for data augmentation
    # then retrain the original models
    print('New data comming', len(ids))

    ensemble_clf = joblib.load('models/ensemble_models/' + dataset_name + '_ensemble.pkl')
    if dataset_name == 'adult':
        protected_attribs = pre_census_income.protected_attribs
        X_train = pre_census_income.X_train_all
        y_train = pre_census_income.y_train_all
        X_test = pre_census_income.X_test
        y_test = pre_census_income.y_test
        model = train_census_income.model
    elif dataset_name == 'german':
        protected_attribs = pre_german_credit.protected_attribs
        X_train = pre_german_credit.X_train
        y_train = pre_german_credit.y_train
        X_test = pre_german_credit.X_test
        y_test = pre_german_credit.y_test
        model = train_german_credit.model
    elif dataset_name == 'bank':
        protected_attribs = pre_bank_marketing.protected_attribs
        X_train = pre_bank_marketing.X_train_all
        y_train = pre_bank_marketing.y_train_all
        X_test = pre_bank_marketing.X_test
        y_test = pre_bank_marketing.y_test
        model = train_bank_marketing.model
    ids_aug = np.empty(shape=(0, len(X_train[0])))

    for x in ids:
        ids_aug = np.append(ids_aug, [x], axis=0)

    #
    # num_aug = int(len(ids) * 0.05)
    # for _ in range(num_aug):
    #     rand_index = np.random.randint(len(ids))
    #     ids_aug = np.append(ids_aug, [ids[rand_index]], axis=0)

    label_vote = ensemble_clf.predict(np.delete(ids_aug, protected_attribs, axis=1))
    X_train = np.append(X_train, ids_aug, axis=0)
    y_train = np.append(y_train, label_vote, axis=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[keras.callbacks.EarlyStopping(patience=10)])
    model.evaluate(X_test, y_test)
    model.save('models/models_from_tests/' + dataset_name + '_' + approach_name + '_retrained_model_fei.h5')




ids_C_g_ADF = np.load('logging_data/logging_data_from_tests/complete_comparison/adult_model.h5_C-g_ids_ADF_1.npy')
ids_C_g_EIDIG_5 = np.load('logging_data/logging_data_from_tests/complete_comparison/adult_model.h5_C-g_ids_EIDIG_5_1.npy')
ids_C_g_EIDIG_INF = np.load('logging_data/logging_data_from_tests/complete_comparison/adult_model.h5_C-g_ids_EIDIG_INF_1.npy')


retraining('adult', 'ADF', ids_C_g_ADF)
retraining('adult', 'EIDIG_5', ids_C_g_EIDIG_5)
retraining('adult', 'EIDIG_INF', ids_C_g_EIDIG_INF)
