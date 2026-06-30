"""
This python file constructs and trains the model for German Credit Dataset.
"""


import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from preprocessing import pre_compas_scores
from tensorflow import keras
# from tensorflow import set_random_seed

import tensorflow as tf
import numpy as np 
np.random.seed(1)
tf.random.set_seed(1)

# create and train a six-layer neural network for the binary classification task
model = keras.Sequential([
    keras.layers.Dense(50, activation="relu", input_shape=pre_compas_scores.X_train.shape[1:]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(15, activation="relu"),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(5, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy","AUC"])


history = model.fit(pre_compas_scores.X_train, pre_compas_scores.y_train, epochs=30)
results = model.evaluate(pre_compas_scores.X_test, pre_compas_scores.y_test) 
print("Evaluation results:", results)
model.save("models/models_from_tests/compas_model.h5")

