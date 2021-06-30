import os
import keras
import numpy as np
import keras.backend as K
import tensorflow as tf
from PIL import Image
from .mobilenet import mobilenet
from keras.layers import Input, Dense, Conv2D, BatchNormalization
from keras.layers import MaxPooling2D, Flatten, Lambda, PReLU, DepthwiseConv2D
from keras.optimizers import SGD
from keras.models import Model


def arcface(input_shape, embedding_size=256):
    inputs = Input(shape=input_shape)
    model = mobilenet(inputs)
    x = keras.layers.Conv2D(512, 1, use_bias=False, name="conv2d")(model.output)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.PReLU(shared_axes=[1, 2])(x)
    
    x = keras.layers.DepthwiseConv2D(int(x.shape[1]), depth_multiplier=1, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(embedding_size, 1, use_bias=False, activation=None)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.BatchNormalization(name="embedding")(x)
    x = Lambda(lambda  x: K.l2_normalize(x, axis=1))(x)
    model = Model(inputs,x)
    return model