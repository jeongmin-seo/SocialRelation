import os
import numpy as np
import scipy.io as sio
import keras.layers
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout


def build_deep_conv_net():
    deep_conv_net = Sequential()

    # layer 1, input size: 48 x 48 x 1
    deep_conv_net.add(Conv2D(64, 5, padding='same', activation='relu', input_shape=(48, 48, 1),
                             kernel_regularizer=keras.regularizers.l2(0.01)))
    deep_conv_net.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    deep_conv_net.add(BatchNormalization())
    deep_conv_net.add(Dropout(0.5))

    # layer 2, input size: 24 x 24 x 64
    deep_conv_net.add(Conv2D(96, 5, padding='same', activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.01)))
    deep_conv_net.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    deep_conv_net.add(BatchNormalization())
    deep_conv_net.add(Dropout(0.5))

    # layer 3, input size: 12 x 12 x 96
    deep_conv_net.add(Conv2D(256, 5, padding='same', activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.01)))
    deep_conv_net.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    deep_conv_net.add(BatchNormalization())
    deep_conv_net.add(Dropout(0.5))

    # layer 4, input size: 6 x 6 x 256
    deep_conv_net.add(Conv2D(256, 5, padding='same', activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.01)))
    deep_conv_net.add(BatchNormalization())
    deep_conv_net.add(Dropout(0.5))

    # layer 5, input size: 6 x 6 x 256
    deep_conv_net.add(Dense(2048, activation='relu',
                            kernel_regularizer=keras.regularizers.l2(0.01)))
    deep_conv_net.add(BatchNormalization())
    deep_conv_net.add(Dropout(0.5))

    # output size: 2048
    return deep_conv_net


def build_social_relation_net():
    deep_conv_net = build_deep_conv_net()



#()()
#('')HAANJU.YOO
