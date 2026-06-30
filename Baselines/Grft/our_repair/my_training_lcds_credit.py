"""
This python file constructs and trains the model for Census Income Dataset.
"""


import sys, os
# sys.path.append("..")
# sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from preprocessing import pre_german_credit
import tensorflow as tf
from tensorflow import keras
import numpy as np
import  itertools
import copy
from tensorflow.keras import backend as K

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def similar_set(X, num_attribs, protected_attribs, constraint):
    # find all similar inputs corresponding to different combinations of protected attributes with non-protected attributes unchanged
    similar_X = []
    protected_domain = []
    for i in protected_attribs:
        protected_domain = protected_domain + [list(range(constraint[i][0], constraint[i][1]+1))]
    all_combs = np.array(list(itertools.product(*protected_domain)))
    # print(all_combs)
    for i, comb in enumerate(all_combs):
        X_new = tf.identity(X)  # Create a copy of the tensor
        for a, c in zip(protected_attribs, comb):
        # Create indices for the column to be updated
            indices = tf.range(tf.shape(X_new)[0])  # Indices for the batch dimension
            indices = tf.expand_dims(indices, 1)  # Reshape to (batch_size, 1) to match the shape requirements
            updates = tf.ones((tf.shape(X_new)[0],), dtype=X.dtype) * c  # Values to update with
            # Combine row indices with column index to create indices for scatter_nd
            scatter_indices = tf.concat([indices, tf.ones_like(indices) * a], axis=1)
            X_new = tf.tensor_scatter_nd_update(X_new, scatter_indices, updates)  # Update tensor
        similar_X.append(X_new)
    return similar_X


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
    total_loss = (loss1+loss2)/2+0.8*difference
    # print(total_loss)
    return total_loss


# for benchmark in ['a','g','g&a']:
for benchmark in ['a']:
    # create and train a six-layer neural network for the binary classification task
    model = keras.Sequential([
        keras.layers.Dense(50, activation="relu", input_shape=pre_german_credit.X_train.shape[1:]),
        keras.layers.Dense(30, activation="relu"),
        keras.layers.Dense(15, activation="relu"),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(5, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    pos_map = { 'a': [9],
                'g': [6],
                'g&a': [6, 9],
                }

    # print(model.summary())
    X_train = pre_german_credit.X_train
    Y_train = pre_german_credit.y_train
    constraints = pre_german_credit.constraint
    num_attribs = len(X_train[0])
    protected_attribs = pos_map[benchmark]

    # 编译模型
    optimizer = keras.optimizers.Nadam()
    loss_fn = keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn,metrics=["accuracy"])
    # 自定义训练步骤
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            loss2 = 0
            difference=0
            # 计算模型的预测值
            y_pred = model(x, training=True)
            # y_true = tf.reshape(y, (-1, 1))
            # y_true = tf.cast(y_true, dtype=tf.float32)
            # loss1 = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            # similar_x = similar_set(x,num_attribs,protected_attribs,constraints)
            # tf.print(similar_x)
            # # 计算原始输入和变异输入的预测差异
            # for i in range(len(similar_x)):
            #     pred_similar_x_i = model(similar_x[i], training=True)
            #     loss2 += K.binary_crossentropy(y_true, pred_similar_x_i)
            #     difference += K.sum(K.abs(y_pred-pred_similar_x_i))
            # loss2 = loss2/float(len(similar_x))
            # difference = difference/float(len(similar_x))
            # loss_value = (loss1+loss2)/2+difference
            loss_value = custom_loss_new(y,y_pred,x)
        # 计算梯度并更新模型参数
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        return loss_value

    # 创建 TensorFlow 数据集
    batch_size = 32
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(batch_size)

    # 自定义训练循环
    epochs = 30
    for epoch in range(epochs):
        print(f'Start of epoch {epoch}')
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            loss_value = train_step(x_batch_train, y_batch_train)
            
            if step % 10 == 0:
                print(f'Epoch {epoch} Step {step} Loss {loss_value.numpy()}')

    # 评估模型
    results = model.evaluate(pre_german_credit.X_test, pre_german_credit.y_test)
    print("Test loss, Test accuracy:", results)

    # 保存模型
    model.save("models/models_lcds/german_model_"+benchmark+".h5")







# model.compile(loss=custom_loss_(X_train[:2],model,num_attribs,protected_attribs,constraints), optimizer="nadam", metrics=["accuracy"])

# # uncomment for training
# """
# history = model.fit(pre_census_income.X_train, pre_census_income.y_train, epochs=30, validation_data=(pre_census_income.X_val, pre_census_income.y_val))
# model.evaluate(pre_census_income.X_test, pre_census_income.y_test) # 84.32% accuracy
# model.save("models/models_adv/adult_model.h5")
# """
# history = model.fit(pre_census_income.X_train[:2], pre_census_income.y_train[:2], epochs=1, validation_data=(pre_census_income.X_val, pre_census_income.y_val))
# model.evaluate(pre_census_income.X_test, pre_census_income.y_test) # 84.32% accuracy
# model.save("models/models_adv/adult_model.h5")
# # The precision rate is  0.7338425381903643 , the recall rate is  0.5454148471615721 , and the F1 score is  0.625751503006012