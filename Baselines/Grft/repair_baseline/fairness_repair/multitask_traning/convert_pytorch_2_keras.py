

import sys , os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
sys.path.append("/data/d/liliquan/fairness/tabular_fairness")
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from tensorflow import keras
import tensorflow as tf 

import numpy as np 
from argparse import ArgumentParser
from tqdm import tqdm 

import torch
import torch.nn as nn  
import torch.nn.functional as F
import torch.optim as optim  
from torch.utils.data import DataLoader


import torchvision 

import itertools
import collections

import random 
torch.manual_seed(42)
# tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# sys.path.append("..")
# sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from preprocessing import pre_bank_marketing
import generation_utilities as generation_util  
from models_torch import MultiTaskModel as Net 
print(pre_bank_marketing.constraint)
classes_list=[1,1,4 ,5]
# 加载模型权重
torch_model= Net(in_channel=12,classes_list=classes_list)
torch_model.load_state_dict(torch.load('/data/d/liliquan/fairness/tabular_fairness/ckp/pre_census_income-epoch_59-acc_0.8380591667519706-fairg_0.01175.pth'))

model_keras = keras.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[None,12]),
    keras.layers.Dense(20, activation="relu"),
    keras.layers.Dense(15, activation="relu"),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(5, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

model_keras.summary()
#
dummy_input = torch.randn(4,12)
dummy_input_np = dummy_input.numpy()
print (dummy_input.mean(),dummy_input.std(),"mean->std")


with torch.no_grad():
    torch_model.eval()
    # dummy_out1,_ = torch_model(dummy_input)
# torch.onnx.export(torch_model, dummy_input, "model.onnx")

tf_out_before = model_keras(dummy_input_np)

keras_convert = torch_model.convert_torch_2_keras(Model_keras=model_keras)
keras_convert.save("models/multitask_models/bank.h5")
tf_out = keras_convert(dummy_input_np)
#
print (dummy_out1, tf_out_before,tf_out)