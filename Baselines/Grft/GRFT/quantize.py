import tensorflow as tf
import argparse
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.mobilenet import MobileNet
# from PIL import Image
from tensorflow.keras import backend as K

from tensorflow.keras.applications.vgg16 import preprocess_input
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

os.sys.path.append('../')

class  ScaleLayer(tf.keras.layers.Layer):
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


def transferModel(model, percent):
    weights = model.get_weights()

    w = np.asarray(weights[0])
    for item in weights[1:]:
        item = np.asarray(item)
        w = np.append(w, item)

    weights_size = np.size(w)
    sample_idxs = np.random.choice(weights_size, int(weights_size * percent), replace=False)

    w[sample_idxs] = np.float16(w[sample_idxs])
    cur = 0
    for index, item  in enumerate(weights):
        shape = item.shape
        size = np.size(item)
        sh = w[cur:cur+size]
        cur += size
        weights[index] = np.reshape(sh, shape)

    model.set_weights(weights)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='coverage guided fuzzing')

    parser.add_argument('-o', help='')
    parser.add_argument('-i')

    parser.add_argument('-percentage', help='The percentage to truncate weights from FLOAT32 to FLOAT16', default=1.0, type=float)
    parser.add_argument('-model_type', help='Out path', default='lsac_model')
    args = parser.parse_args()

    if args.model_type == 'mobilenet':
        img_rows, img_cols = 224, 224
        input_shape = (img_rows, img_cols, 3)
        input_tensor = Input(shape=input_shape)
        model2 = MobileNet(input_tensor=input_tensor)
    elif args.model_type == 'resnet50':
        img_rows, img_cols = 224, 224
        input_shape = (img_rows, img_cols, 3)
        input_tensor = Input(shape=input_shape)
        model2 = ResNet50(input_tensor=input_tensor)
    else:
        model1= keras.models.load_model(args.i,custom_objects={'ScaleLayer': ScaleLayer})

    if not os.path.exists(args.o):
        os.makedirs(args.o)
    percentage = [0.01, 0.5, 1]
    for p in percentage:
        model = transferModel(model1, p)
        save_model(model, os.path.join(args.o, str(args.model_type) +'_'+str(p)+'.h5'))
        

