
from tensorflow.keras import backend as K
import tensorflow as tf
class  ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, dense_len, min=-1, max=1, num=0, **kwargs):
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
