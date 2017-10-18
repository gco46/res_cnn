# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 12:08:27 2016

@author: tanaka
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Activation, Input, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, ZeroPadding2D
from keras.layers.merge import Add
from keras.initializers import Constant
from keras.regularizers import l2
from keras.engine.topology import Layer
import keras.backend as K
import numpy as np
from keras.utils import conv_utils
from keras.engine.topology import Layer
from keras.engine import InputSpec


def bilinear_upsample_weights(factor, n_class, out_ch):
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
    weights = np.zeros((filter_size, filter_size, n_class, out_ch),
                       dtype=np.float32)

    weights[:, :, :, :] = upsample_kernel[:, :, None, None]
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


def FCN_8s_norm(classes, in_shape, l2_reg, nopad=False):
    """
    VGG16 based FCN model,
    classes: int, number of classes

    return: keras Model object
    """
    inputs = Input(shape=in_shape)
    if nopad:
        x = inputs
    else:
        x = ZeroPadding2D(padding=(100, 100))(inputs)
    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same',
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same',
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=128,
               kernel_size=(3, 3),
               padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=128,
               kernel_size=(3, 3),
               padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=256,
               kernel_size=(3, 3),
               padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=256,
               kernel_size=(3, 3),
               padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=256,
               kernel_size=(3, 3),
               padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    # pool3のfeature mapを取得
    p3 = x

    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    # pool4のfeature mapを取得
    p4 = x

    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=4096,
               kernel_size=(7, 7),
               padding="valid",
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)
    x = Conv2D(filters=4096,
               kernel_size=(1, 1),
               padding="valid",
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)

    if nopad:
        p5 = Conv2DTranspose(filters=classes,
                             kernel_size=(14, 14),
                             strides=(1, 1),
                             padding="valid",
                             activation="linear",
                             kernel_regularizer=l2(l2_reg),
                             kernel_initializer=Constant(
                                 bilinear_upsample_weights(
                                     "full", classes, 4096)
                             ))(x)
    else:
        p5 = Conv2DTranspose(filters=classes,
                             kernel_size=(4, 4),
                             strides=(2, 2),
                             padding="same",
                             activation="linear",
                             kernel_regularizer=l2(l2_reg),
                             kernel_initializer=Constant(
                                 bilinear_upsample_weights(2, classes, 4096)
                             ))(x)

    # pool3 のfeature mapを次元圧縮
    p3 = Conv2D(filters=classes,
                kernel_size=(1, 1),
                kernel_regularizer=l2(l2_reg),
                activation='relu')(p3)
    # pool4のfeature mapを次元圧縮
    p4 = Conv2D(filters=classes,
                kernel_size=(1, 1),
                kernel_regularizer=l2(l2_reg),
                activation="relu")(p4)

    # merge p4 and p5
    p4 = CroppingLike2D(K.int_shape(p5))(p4)
    p45 = Add()([p4, p5])

    # p4+p5 を x2 upsampling
    if not nopad:
        p45 = ZeroPadding2D(padding=(1, 1))(p45)
    p45 = Conv2DTranspose(filters=classes,
                          kernel_size=(4, 4),
                          strides=(2, 2),
                          padding="same",
                          activation="linear",
                          kernel_regularizer=l2(l2_reg),
                          kernel_initializer=Constant(
                              bilinear_upsample_weights(2, classes, classes)
                          ))(p45)

    # p3とp45をmerge
    p3 = CroppingLike2D(K.int_shape(p45))(p3)
    p345 = Add()([p3, p45])

    # p3+p4+p5を x8 upsampling
    if not nopad:
        p345 = ZeroPadding2D(padding=(1, 1))(p345)
    x = Conv2DTranspose(filters=classes,
                        kernel_size=(16, 16),
                        strides=(8, 8),
                        padding="same",
                        activation="linear",
                        kernel_regularizer=l2(l2_reg),
                        kernel_initializer=Constant(
                            bilinear_upsample_weights(8, classes, classes)
                        ))(p345)

    x = CroppingLike2D(K.int_shape(inputs))(x)
    model = Model(inputs=inputs, outputs=x)
    return model


def FCN_8s(classes, in_shape, l2_reg, nopad=False):
    """
    VGG16 based FCN model,
    classes: int, number of classes

    return: keras Model object
    """
    inputs = Input(shape=in_shape)
    if nopad:
        x = inputs
    else:
        x = ZeroPadding2D(padding=(100, 100))(inputs)
    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same',
               kernel_regularizer=l2(l2_reg))(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same',
               kernel_regularizer=l2(l2_reg))(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=128,
               kernel_size=(3, 3),
               padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=128,
               kernel_size=(3, 3),
               padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=256,
               kernel_size=(3, 3),
               padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=256,
               kernel_size=(3, 3),
               padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=256,
               kernel_size=(3, 3),
               padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    # pool3のfeature mapを取得
    p3 = x

    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    # pool4のfeature mapを取得
    p4 = x

    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=4096,
               kernel_size=(7, 7),
               padding="valid",
               kernel_regularizer=l2(l2_reg))(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)
    x = Conv2D(filters=4096,
               kernel_size=(1, 1),
               padding="valid",
               kernel_regularizer=l2(l2_reg))(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)

    if nopad:
        p5 = Conv2DTranspose(filters=classes,
                             kernel_size=(14, 14),
                             strides=(1, 1),
                             padding="valid",
                             activation="linear",
                             kernel_regularizer=l2(l2_reg),
                             kernel_initializer=Constant(
                                 bilinear_upsample_weights(
                                     "full", classes, 4096)
                             ))(x)
    else:
        p5 = Conv2DTranspose(filters=classes,
                             kernel_size=(4, 4),
                             strides=(2, 2),
                             padding="same",
                             activation="linear",
                             kernel_regularizer=l2(l2_reg),
                             kernel_initializer=Constant(
                                 bilinear_upsample_weights(2, classes, 4096)
                             ))(x)

    # pool3 のfeature mapを次元圧縮
    p3 = Conv2D(filters=classes,
                kernel_size=(1, 1),
                kernel_regularizer=l2(l2_reg),
                activation='relu')(p3)
    # pool4のfeature mapを次元圧縮
    p4 = Conv2D(filters=classes,
                kernel_size=(1, 1),
                kernel_regularizer=l2(l2_reg),
                activation="relu")(p4)

    # merge p4 and p5
    p4 = CroppingLike2D(K.int_shape(p5))(p4)
    p45 = Add()([p4, p5])

    # p4+p5 を x2 upsampling
    if not nopad:
        p45 = ZeroPadding2D(padding=(1, 1))(p45)
    p45 = Conv2DTranspose(filters=classes,
                          kernel_size=(4, 4),
                          strides=(2, 2),
                          padding="same",
                          activation="linear",
                          kernel_regularizer=l2(l2_reg),
                          kernel_initializer=Constant(
                              bilinear_upsample_weights(2, classes, classes)
                          ))(p45)

    # p3とp45をmerge
    p3 = CroppingLike2D(K.int_shape(p45))(p3)
    p345 = Add()([p3, p45])

    # p3+p4+p5を x8 upsampling
    if not nopad:
        p345 = ZeroPadding2D(padding=(1, 1))(p345)
    x = Conv2DTranspose(filters=classes,
                        kernel_size=(16, 16),
                        strides=(8, 8),
                        padding="same",
                        activation="linear",
                        kernel_regularizer=l2(l2_reg),
                        kernel_initializer=Constant(
                            bilinear_upsample_weights(8, classes, classes)
                        ))(p345)

    x = CroppingLike2D(K.int_shape(inputs))(x)
    model = Model(inputs=inputs, outputs=x)
    return model


class CroppingLike2D(Layer):

    def __init__(self, target_shape, offset=None, data_format=None,
                 **kwargs):
        """Crop to target.
        If only one `offset` is set, then all dimensions are offset by this amount.
        """
        super(CroppingLike2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.target_shape = target_shape
        if offset is None or offset == 'centered':
            self.offset = 'centered'
        elif isinstance(offset, int):
            self.offset = (offset, offset)
        elif hasattr(offset, '__len__'):
            if len(offset) != 2:
                raise ValueError('`offset` should have two elements. '
                                 'Found: ' + str(offset))
            self.offset = offset
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0],
                    input_shape[1],
                    self.target_shape[2],
                    self.target_shape[3])
        else:
            return (input_shape[0],
                    self.target_shape[1],
                    self.target_shape[2],
                    input_shape[3])

    def call(self, inputs):
        input_shape = K.int_shape(inputs)
        if self.data_format == 'channels_first':
            input_height = input_shape[2]
            input_width = input_shape[3]
            target_height = self.target_shape[2]
            target_width = self.target_shape[3]
            if target_height > input_height or target_width > input_width:
                raise ValueError('The Tensor to be cropped need to be smaller'
                                 'or equal to the target Tensor.')

            if self.offset == 'centered':
                self.offset = [int((input_height - target_height) / 2),
                               int((input_width - target_width) / 2)]

            if self.offset[0] + target_height > input_height:
                raise ValueError('Height index out of range: '
                                 + str(self.offset[0] + target_height))
            if self.offset[1] + target_width > input_width:
                raise ValueError('Width index out of range:'
                                 + str(self.offset[1] + target_width))

            return inputs[:,
                          :,
                          self.offset[0]:self.offset[0] + target_height,
                          self.offset[1]:self.offset[1] + target_width]
        elif self.data_format == 'channels_last':
            input_height = input_shape[1]
            input_width = input_shape[2]
            target_height = self.target_shape[1]
            target_width = self.target_shape[2]
            if target_height > input_height or target_width > input_width:
                raise ValueError('The Tensor to be cropped need to be smaller'
                                 'or equal to the target Tensor.')

            if self.offset == 'centered':
                self.offset = [int((input_height - target_height) / 2),
                               int((input_width - target_width) / 2)]

            if self.offset[0] + target_height > input_height:
                raise ValueError('Height index out of range: '
                                 + str(self.offset[0] + target_height))
            if self.offset[1] + target_width > input_width:
                raise ValueError('Width index out of range:'
                                 + str(self.offset[1] + target_width))
            output = inputs[:,
                            self.offset[0]:self.offset[0] + target_height,
                            self.offset[1]:self.offset[1] + target_width,
                            :]
            return output

    def get_config(self):
        config = {'target_shape': self.target_shape,
                  'offset': self.offset,
                  'data_format': self.data_format}
        base_config = super(CroppingLike2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
    log_softmax = y_pred - K.logsumexp(y_pred, axis=1, keepdims=True)
    # log_softmax = tf.nn.log_softmax(y_pred)

    # y_trueをベクトル化、ignore labelを含めたone-hot表現に直す
    # shape = (nb_samples, num_class + 1)
    y_true = K.one_hot(K.cast((K.flatten(y_true)), dtype="int32"),
                       K.int_shape(y_pred)[-1] + 1)
    # class axisで分解し、(nb_samples, 1)のベクトルをリストに格納する
    # その後最後のラベル(ignore label)のベクトルを除去して新たなone-hot行列を得る
    # shape = (nb_samples, num_class)
    y_true = y_true[:, :-1]
    # unpacked = tf.unstack(y_true, axis=-1)
    # y_true = tf.stack(unpacked[:-1], axis=-1)

    # closs entropyを計算し、ピクセル毎の和を算出した後
    # 全ピクセルの和を計算する
    # 平均でもいいか...?
    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy = K.mean(cross_entropy)

    return cross_entropy


def sparse_accuracy(y_true, y_pred):
    """
    define accuracy for fcn, ignoring last label.
    y_true: array, (None, )
    y_pred: array, (None, rows, columns, channels)
    """
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))

    y_true = K.one_hot(K.cast((K.flatten(y_true)), dtype="int32"),
                       nb_classes + 1)

    legal_labels = ~K.cast(y_true[:, -1], "bool")
    y_true = y_true[:, :-1]
    # unpacked = tf.unstack(y_true, axis=-1)
    # legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    # y_true = tf.stack(unpacked[:-1], axis=-1)

    result = K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1))
    score = K.sum(K.cast(legal_labels & result, "float32"))
    total = K.sum(K.cast(legal_labels, "float32"))
    return score / total

    # return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true,
    # axis=-1), K.argmax(y_pred, axis=-1)))) /
    # K.sum(tf.to_float(legal_labels))
