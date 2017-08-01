# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 12:08:27 2016

@author: tanaka
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Activation, Input
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, ZeroPadding2D
from keras.layers.merge import Add
from keras.initializers import Constant
from keras.regularizers import l2
from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf
# from keras.layers.normalization import BatchNormalization
import numpy as np


def bilinear_upsample_weights(factor, number_of_classes):
    if factor == "full":
        filter_size = 14
    else:
        filter_size = factor * 2 - factor % 2
    factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:filter_size, :filter_size]
    upsample_kernel = (1 - abs(og[0] - center) /
                       factor) * (1 - abs(og[1] - center) / factor)
    weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
                       dtype=np.float32)
    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights


def myVGG_p4(size, l2_reg, method, out_num):
    if method == "classification":
        out_act = "softmax"
    elif method == "regression":
        out_act = "linear"
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(size, size, 3)))
    model.add(Conv2D(
        64, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        64, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        128, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        128, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        256, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        256, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        256, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        512, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        512, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        512, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(0.5))
    model.add(Dense(out_num, activation=out_act,
                    kernel_regularizer=l2(l2_reg)))

    return model


def myVGG_p5(size, l2_reg, method, out_num):
    """
    this model is same to VGG16
    """
    if method == "classification":
        out_act = "softmax"
    elif method == "regression":
        out_act = "linear"
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(size, size, 3)))
    model.add(Conv2D(
        64, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        64, (3, 3), activation="relu", kernel_regularizer=l2(l2_reg)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        128, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        128, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        256, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        256, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        256, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        512, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        512, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        512, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        512, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        512, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        512, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(0.5))

    model.add(Dense(4096, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(0.5))

    model.add(Dense(out_num, activation=out_act,
                    kernel_regularizer=l2(l2_reg)))

    return model


def fcn_p5_full(classes):
    """
    VGG16 based FCN model,
    classes: int, number of classes

    return: keras Model object
    """
    inputs = Input(shape=(224, 224, 3))
    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same',
               activation='relu')(inputs)
    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same',
               activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=128,
               kernel_size=(3, 3),
               padding="same",
               activation='relu')(x)
    x = Conv2D(filters=128,
               kernel_size=(3, 3),
               padding="same",
               activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=256,
               kernel_size=(3, 3),
               padding="same",
               activation='relu')(x)
    x = Conv2D(filters=256,
               kernel_size=(3, 3),
               padding="same",
               activation='relu')(x)
    x = Conv2D(filters=256,
               kernel_size=(3, 3),
               padding="same",
               activation='relu')(x)
    x = MaxPooling2D()(x)

    # pool3のfeature mapを取得
    p3 = x

    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               activation='relu')(x)
    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               activation='relu')(x)
    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               activation='relu')(x)
    x = MaxPooling2D()(x)

    # pool4のfeature mapを取得
    p4 = x

    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               activation='relu')(x)
    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               activation='relu')(x)
    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=4096,
               kernel_size=(7, 7),
               padding="valid",
               activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Conv2D(filters=4096,
               kernel_size=(1, 1),
               padding="valid",
               activation="relu")(x)
    x = Dropout(0.5)(x)

    p5 = Conv2DTranspose(filters=classes,
                         kernel_size=(14, 14),
                         strides=(1, 1),
                         padding="valid",
                         activation="linear",
                         kernel_initializer=Constant(bilinear_upsample_weights("full", classes)))(x)

    # pool3 のfeature mapを次元圧縮
    p3 = Conv2D(filters=classes,
                kernel_size=(1, 1),
                activation='relu')(p3)
    # pool4のfeature mapを次元圧縮
    p4 = Conv2D(filters=classes,
                kernel_size=(1, 1),
                activation="relu")(p4)

    # merge p4 and p5
    p45 = Add()([p4, p5])

    # p4+p5 を x2 upsampling
    p45 = Conv2DTranspose(filters=classes,
                          kernel_size=(4, 4),
                          strides=(2, 2),
                          padding="same",
                          activation="linear",
                          kernel_initializer=Constant(bilinear_upsample_weights(2, classes)))(p45)

    # p3とp45をmerge
    p345 = Add()([p3, p45])

    # p3+p4+p5を x8 upsampling
    x = Conv2DTranspose(filters=classes,
                        kernel_size=(16, 16),
                        strides=(8, 8),
                        padding="same",
                        activation="linear",
                        kernel_initializer=Constant(bilinear_upsample_weights(8, classes)))(p345)

    model = Model(inputs=inputs, outputs=x)
    return model


def fcn_p5_image(classes):
    """
    VGG16 based FCN model,
    classes: int, number of classes

    return: keras Model object
    """
    inputs = Input(shape=(None, None, 3))
    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same',
               activation='relu')(inputs)
    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same',
               activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=128,
               kernel_size=(3, 3),
               padding="same",
               activation='relu')(x)
    x = Conv2D(filters=128,
               kernel_size=(3, 3),
               padding="same",
               activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=256,
               kernel_size=(3, 3),
               padding="same",
               activation='relu')(x)
    x = Conv2D(filters=256,
               kernel_size=(3, 3),
               padding="same",
               activation='relu')(x)
    x = Conv2D(filters=256,
               kernel_size=(3, 3),
               padding="same",
               activation='relu')(x)
    x = MaxPooling2D()(x)

    # pool3のfeature mapを取得
    p3 = x

    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               activation='relu')(x)
    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               activation='relu')(x)
    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               activation='relu')(x)
    x = MaxPooling2D()(x)

    # pool4のfeature mapを取得
    p4 = x

    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               activation='relu')(x)
    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               activation='relu')(x)
    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=4096,
               kernel_size=(7, 7),
               padding="valid",
               activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Conv2D(filters=4096,
               kernel_size=(1, 1),
               padding="valid",
               activation="relu")(x)
    x = Dropout(0.5)(x)

    p5 = Conv2DTranspose(filters=classes,
                         kernel_size=(14, 14),
                         strides=(1, 1),
                         padding="valid",
                         activation="linear",
                         kernel_initializer=Constant(bilinear_upsample_weights("full", classes)))(x)

    # pool3 のfeature mapを次元圧縮
    p3 = Conv2D(filters=classes,
                kernel_size=(1, 1),
                activation='relu')(p3)
    # pool4のfeature mapを次元圧縮
    p4 = Conv2D(filters=classes,
                kernel_size=(1, 1),
                activation="relu")(p4)

    # merge p4 and p5
    p45 = Add()([p4, p5])

    # p4+p5 を x2 upsampling
    p45 = Conv2DTranspose(filters=classes,
                          kernel_size=(4, 4),
                          strides=(2, 2),
                          padding="same",
                          activation="linear",
                          kernel_initializer=Constant(bilinear_upsample_weights(2, classes)))(p45)

    # p3とp45をmerge
    p345 = Add()([p3, p45])

    # p3+p4+p5を x8 upsampling
    x = Conv2DTranspose(filters=classes,
                        kernel_size=(16, 16),
                        strides=(8, 8),
                        padding="same",
                        activation="linear",
                        kernel_initializer=Constant(bilinear_upsample_weights(8, classes)))(p345)

    model = Model(inputs=inputs, outputs=x)
    return model


def softmax_sparse_crossentropy(y_true, y_pred):
    """
    define loss function, categorical_crossentropy for fcn.
    ignore the last number label.

    !!!  if you use this function,
        the output layers activation must be linear !!!

    y_true: array, the shape is (None, )
    y_pred: array, the output of fcn, (None, rows, columns, channels)
    """
    # y_predをベクトル化してsoftmaxかける(ピクセル毎に独立として扱う)
    # shape = (nb_samples, num_class)
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    # y_trueをベクトル化、ignore labelを含めたone-hot表現に直す
    # shape = (nb_samples, num_class + 1)
    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                       K.int_shape(y_pred)[-1] + 1)
    # class axisで分解し、(nb_samples, 1)のベクトルをリストに格納する
    # その後最後のラベル(ignore label)のベクトルを除去して新たなone-hot行列を得る
    # shape = (nb_samples, num_class)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    # closs entropyを計算し、ピクセル毎の和を算出した後
    # 全ピクセルの和を計算する
    # 平均でもいいか...?
    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean


def sparse_accuracy(y_true, y_pred):
    """
    define accuracy for fcn, ignoring last label.
    y_true: array, (None, )
    y_pred: array, (None, rows, columns, channels)
    """
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                       nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))
