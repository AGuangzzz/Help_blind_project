import os
import cv2
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, Reshape, Activation, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.applications.imagenet_utils import preprocess_input
from keras.backend import set_session
from keras.models import Model

from .backbone_retinaface import MobileNet, UpsampleLike, ResNet50
from .utils import compose
# ---------------------------------------------------#
#   卷积块
#   Conv2D + BatchNormalization + LeakyReLU
# ---------------------------------------------------#
def Conv2D_BN_Leaky(*args, **kwargs):
    leaky = 0.1
    try:
        leaky = kwargs["leaky"]
        del kwargs["leaky"]
    except:
        pass
    return compose(
        Conv2D(*args, **kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=leaky))

# ---------------------------------------------------#
#   卷积块
#   Conv2D + BatchNormalization + LeakyReLU
# ---------------------------------------------------#
def Conv2D_BN(*args, **kwargs):
    return compose(
        Conv2D(*args, **kwargs),
        BatchNormalization())


def SSH(inputs, out_channel, leaky=0.1):
    conv3X3 = Conv2D_BN(out_channel // 2, kernel_size=3, strides=1, padding='same')(inputs)

    conv5X5_1 = Conv2D_BN_Leaky(out_channel // 4, kernel_size=3, strides=1, padding='same', leaky=leaky)(inputs)
    conv5X5 = Conv2D_BN(out_channel // 4, kernel_size=3, strides=1, padding='same')(conv5X5_1)

    conv7X7_2 = Conv2D_BN_Leaky(out_channel // 4, kernel_size=3, strides=1, padding='same', leaky=leaky)(conv5X5_1)
    conv7X7 = Conv2D_BN(out_channel // 4, kernel_size=3, strides=1, padding='same')(conv7X7_2)

    out = Concatenate(axis=-1)([conv3X3, conv5X5, conv7X7])
    out = Activation("relu")(out)
    return out


def ClassHead(inputs, num_anchors=2):
    outputs = Conv2D(num_anchors * 2, kernel_size=1, strides=1)(inputs)
    return Activation("softmax")(Reshape([-1, 2])(outputs))


def BboxHead(inputs, num_anchors=2):
    outputs = Conv2D(num_anchors * 4, kernel_size=1, strides=1)(inputs)
    return Reshape([-1, 4])(outputs)


def LandmarkHead(inputs, num_anchors=2):
    outputs = Conv2D(num_anchors * 5 * 2, kernel_size=1, strides=1)(inputs)
    return Reshape([-1, 10])(outputs)


def RetinaFace(cfg, backbone="mobilenet"):
    inputs = Input(shape=(None, None, 3))

    if backbone == "mobilenet":
        C3, C4, C5 = MobileNet(inputs)
    elif backbone == "resnet50":
        C3, C4, C5 = ResNet50(inputs)
    else:
        raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))

    leaky = 0
    if (cfg['out_channel'] <= 64):
        leaky = 0.1
    P3 = Conv2D_BN_Leaky(cfg['out_channel'], kernel_size=1, strides=1, padding='same', name='C3_reduced', leaky=leaky)(
        C3)
    P4 = Conv2D_BN_Leaky(cfg['out_channel'], kernel_size=1, strides=1, padding='same', name='C4_reduced', leaky=leaky)(
        C4)
    P5 = Conv2D_BN_Leaky(cfg['out_channel'], kernel_size=1, strides=1, padding='same', name='C5_reduced', leaky=leaky)(
        C5)

    P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, P4])
    P4 = Add(name='P4_merged')([P5_upsampled, P4])
    P4 = Conv2D_BN_Leaky(cfg['out_channel'], kernel_size=3, strides=1, padding='same', name='Conv_P4_merged',
                         leaky=leaky)(P4)

    P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, P3])
    P3 = Add(name='P3_merged')([P4_upsampled, P3])
    P3 = Conv2D_BN_Leaky(cfg['out_channel'], kernel_size=3, strides=1, padding='same', name='Conv_P3_merged',
                         leaky=leaky)(P3)

    SSH1 = SSH(P3, cfg['out_channel'], leaky=leaky)
    SSH2 = SSH(P4, cfg['out_channel'], leaky=leaky)
    SSH3 = SSH(P5, cfg['out_channel'], leaky=leaky)

    SSH_all = [SSH1, SSH2, SSH3]

    bbox_regressions = Concatenate(axis=1, name="bbox_reg")([BboxHead(feature) for feature in SSH_all])
    classifications = Concatenate(axis=1, name="cls")([ClassHead(feature) for feature in SSH_all])
    ldm_regressions = Concatenate(axis=1, name="ldm_reg")([LandmarkHead(feature) for feature in SSH_all])

    output = [bbox_regressions, classifications, ldm_regressions]

    model = Model(inputs=inputs, outputs=output)
    return model
