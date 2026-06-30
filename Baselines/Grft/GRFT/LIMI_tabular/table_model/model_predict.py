import copy
import math
import os
import numpy as np
import sys
import pickle
import argparse
# sys.path.append("/data/c/liliquan/fairness_testing/Latent_Imitator/LIMI_tabular/data")
sys.path.append("/data/c/liliquan/fairness_testing/Latent_Imitator/LIMI_tabular")
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.platform import flags
from data.census import census_predict_data
from data.bank import bank_predict_data
from data.credit import credit_predict_data
from data.compas import compas_predict_data
from data.lsac import lsac_predict_data
# from data.meps import meps_predict_data
# from utils.utils_tf import model_train, model_eval
# from table_model.dnn_models import dnn
# import tensorflow as tf
# 确保启用 Eager Execution
import tensorflow.keras.backend as K
import argparse
import random

# seed(1)
tf.random.set_seed(2)
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# config = tf.ConfigProto()  
# config.gpu_options.allow_growth=True 
# sess = tf.Session(config=config)

# K.set_session(sess)
tf.config.experimental_run_functions_eagerly(True)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# FLAGS = flags.FLAGS
class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, dense_len, min=-1, max=1, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
        tf.keras.constraints.MinMaxNorm()
        self.scale = K.variable([[1. for x in range(dense_len)]], name='ffff',
                                constraint=lambda t: tf.clip_by_value(t, min, max))
        self.dense_len = dense_len
    def call(self, inputs, **kwargs):
        m = inputs * self.scale
        return m
    def get_config(self):
        config = {'dense_len': self.dense_len}
        base_config = super(ScaleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# 自定义损失函数
def custom_loss_new(y_true,y_pred,x):
    loss2=0
    difference=0
    y_true = tf.reshape(y_true, (-1, 1))
    y_true = tf.cast(y_true, dtype=tf.float32)
    loss1 = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    similar_x = similar_set(x,num_attribs,protected_attribs,constraints)
    # tf.print(similar_x)

    # 计算原始输入和变异输入的预测差异
    for i in range(len(similar_x)):
        pred_similar_x_i = model(similar_x[i], training=True)
        loss2 += tf.keras.losses.binary_crossentropy(y_true, pred_similar_x_i)
        difference += K.sum(K.abs(y_pred-pred_similar_x_i))
    loss2 = loss2/float(len(similar_x))
    difference = difference/float(len(similar_x))
    total_loss = (loss1+loss2)/2+difference
    # print(total_loss)
    return total_loss


# def predicting(dataset, dataset_path, model_path, batch_size):
#     """
#     Train the model
#     :param dataset: the name of testing dataset
#     :param model_path: the path to save trained model
#     """
#     data = {
#         "census": census_predict_data,
#         "credit": credit_predict_data,
#         "bank": bank_predict_data,
#         "compas": compas_predict_data,
#         "meps": meps_predict_data,
#     }
#     # prepare the data and model
#     X, Y, input_shape, nb_classes = data[dataset](dataset_path)
#     tf.set_random_seed(1234)
#     config = tf.ConfigProto()
#     # config.gpu_options.per_process_gpu_memory_fraction = 0.8
#     sess = tf.Session(config=config)
#     x = tf.placeholder(tf.float32, shape=input_shape)
#     y = tf.placeholder(tf.float32, shape=(None, nb_classes))
#     model = dnn(input_shape, nb_classes)
#     preds = model(x)
#     saver = tf.train.Saver()
#     model_path = os.path.join(model_path, "dnn", "best.model")
#     print("load model from ", model_path)
#     saver.restore(sess, model_path)

#     with sess.as_default():
#         nb_batches = int(math.ceil(float(len(X)) / batch_size))
#         assert nb_batches * batch_size >= len(X)
#         pros_all = np.zeros(shape=(X.shape[0], nb_classes), dtype="float32")

#         X_cur = np.zeros((batch_size,) + X.shape[1:], dtype=X.dtype)
#         for batch in range(nb_batches):
#             if batch % 100 == 0 and batch > 0:
#                 print("Batch " + str(batch))
#             start = batch * batch_size
#             end = min(len(X), start + batch_size)
#             cur_batch_size = end - start
#             X_cur[:cur_batch_size] = X[start:end]
#             # feed_dict = {x: X_cur}
#             feed_dict = {x: X[start:end]}
#             pros = sess.run(preds, feed_dict)
#             for i in range(start, end):
#                 pros_all[i] = pros[i - start]
#     print("X[0]", X[0])
#     print("len(pros_all)", len(pros_all))
#     print("pros_all[0]", pros_all[0])
#     labels = np.argmax(pros_all, axis=1)
#     print("labels[0]", labels[0])
#     return pros_all, labels

def predicting(dataset, dataset_path, model_path, batch_size):
    """
    Train the model
    :param dataset: the name of testing dataset
    :param model_path: the path to save trained model
    """
    data = {
        "census": census_predict_data,
        "credit": credit_predict_data,
        "bank": bank_predict_data,
        "compas": compas_predict_data,
        "lsac": lsac_predict_data,
    }
    # prepare the data and model
    X, Y, input_shape, nb_classes = data[dataset](dataset_path)
    loss_fn = keras.losses.BinaryCrossentropy()
    # model = keras.models.load_model(model_path,custom_objects={'ScaleLayer': ScaleLayer})
    model = keras.models.load_model(model_path,custom_objects={'ScaleLayer': ScaleLayer,'loss':custom_loss_new})
    tf.random.set_seed(1234)
    # config = tf.ConfigProto()
    print(X[0])
    nb_batches = int(math.ceil(float(len(X)) / batch_size))
    assert nb_batches * batch_size >= len(X)
    pros_all = np.zeros(shape=(X.shape[0], nb_classes), dtype="float32")

    X_cur = np.zeros((batch_size,) + X.shape[1:], dtype=X.dtype)
    for batch in range(nb_batches):
        if batch % 100 == 0 and batch > 0:
            print("Batch " + str(batch))
        start = batch * batch_size
        end = min(len(X), start + batch_size)
        cur_batch_size = end - start
        X_cur[:cur_batch_size] = X[start:end]
        # feed_dict = {x: X_cur}
        # feed_dict = {x: X[start:end]}
        x = X[start:end]
        pros = model.predict(x)

        # with tf.Session() as sess:
        #     pros = sess.run(pros)
        for i in range(start, end):
            pros_all[i] = pros[i - start]
    print("X[0]", X[0])
    print("len(pros_all)", len(pros_all))
    print("pros_all[0]", pros_all[0])
    # labels = np.argmax(pros_all, axis=1)
    labels = (pros_all>0.5).astype('int').flatten()
    print("labels[0]", labels[0])
    return pros_all, labels


def main(argv=None):
    pros_all, labels = predicting(
        dataset=FLAGS.dataset,
        dataset_path=FLAGS.dataset_path,
        model_path=FLAGS.model_path,
        batch_size=FLAGS.batch_size,
    )

    with open(FLAGS.output_path, "wb+") as handle:
        pickle.dump(pros_all, handle)
    with open(FLAGS.output_path2, "wb+") as handle:
        pickle.dump(labels, handle)


if __name__ == "__main__":
    # flags.DEFINE_string("dataset", "census", "the name of dataset")
    # flags.DEFINE_string(
    #     "dataset_path", "../datasets/census", "the path of test dataset"
    # )

    # flags.DEFINE_string(
    #     "model_path", "../logs/census", "the name of path for saving model"
    # )
    # flags.DEFINE_integer("batch_size", 64, "Size of training batches")
    # flags.DEFINE_string(
    #     "output_path", "../logs/census/predict_scores.npy", "Size of training batches"
    # )
    # flags.DEFINE_string(
    #     "output_path2", "../logs/census/labels.npy", "Size of training batches"
    # )
    # tf.app.run()
    parser = argparse.ArgumentParser(description='model predict')
    parser.add_argument('--dataset', default='census', help='dataset')
    parser.add_argument('--dataset_path', default='../datasets/census', help='dataset_path')
    parser.add_argument('--model_path', default='../logs/census', help='the name of path for saving model')
    parser.add_argument('--batch_size', type=int,default=64, help='dataset_path')
    parser.add_argument('--output_path',default='../logs/census/predict_scores.npy', help='dataset_path')
    parser.add_argument('--output_path2',default='../logs/census/labels.npy', help='dataset_path')
    args = parser.parse_args()
    pros_all, labels = predicting(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
    )
    directory_path = os.path.dirname(args.output_path)
    if directory_path and not os.path.exists(directory_path):
        os.makedirs(directory_path)
    with open(args.output_path, "wb+") as handle:
        pickle.dump(pros_all, handle)
    directory_path = os.path.dirname(args.output_path2)
    if directory_path and not os.path.exists(directory_path):
        os.makedirs(directory_path)
    with open(args.output_path2, "wb+") as handle:
        pickle.dump(labels, handle)