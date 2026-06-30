"""
This python file constructs and trains the model for German Credit Dataset.
"""


import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from preprocessing import pre_lsac 
import tensorflow as tf
from tensorflow import keras
# from tensorflow import set_random_seed
# from numpy.random import seed

import numpy as np 
np.random.seed(1)
tf.random.set_seed(1)

# create and train a six-layer neural network for the binary classification task
model = keras.Sequential([
    keras.layers.Dense(50, activation="relu", input_shape=pre_lsac.X_train.shape[1:], name="layer1"),
    keras.layers.Dense(30, activation="relu", name="layer2"),
    keras.layers.Dense(15, activation="relu", name="layer3"),
    keras.layers.Dense(10, activation="relu", name="layer4"),
    keras.layers.Dense(5, activation="relu", name="layer5"),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy","AUC"])

history = model.fit(pre_lsac.X_train, pre_lsac.y_train, epochs=30)
model.evaluate(pre_lsac.X_test, pre_lsac.y_test) 
model.save("models/lsac_model.h5")

