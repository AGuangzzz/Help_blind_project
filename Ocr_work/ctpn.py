import keras
from keras import layers
from keras import Input, Model
import tensorflow as tf
from .resnet50 import resnet50


def ctpn_net(inputs_shape, rnn_units = 64, fc_units = 256, num_anchors = 10):
    # 网络构建
    input_image = Input(shape=inputs_shape, name='input_image')

    #-------------------------------------------------#
    #   主干网络构建
    #-------------------------------------------------#
    base_features = resnet50(input_image)

    #-------------------------------------------------#
    #   构建ctpn网络
    #-------------------------------------------------#
    # 3x3卷积整合特征
    x = layers.Conv2D(512, kernel_size=(3, 3), padding='same', name='pre_fc')(base_features)  # [B,H,W,512]
    # 沿着宽度方式做rnn
    rnn_forward = layers.TimeDistributed(layers.GRU(rnn_units, return_sequences=True, kernel_initializer='he_normal'),
                                        name='gru_forward')(x)
    rnn_backward = layers.TimeDistributed(layers.GRU(rnn_units, return_sequences=True, kernel_initializer='he_normal', go_backwards=True),
                                        name='gru_backward')(x)
    # 进行堆叠
    rnn_output = layers.Concatenate(name='gru_concat')([rnn_forward, rnn_backward])  # (B,H,W,256)

    # 利用卷积模拟全连接层
    fc_output = layers.Conv2D(fc_units, kernel_size=(1, 1), activation='relu', name='fc_output')(rnn_output)

    # 分类head
    class_logits = layers.Conv2D(2 * num_anchors, kernel_size=(1, 1), name='cls')(fc_output)
    class_logits = layers.Reshape(target_shape=(-1, 2), name='cls_reshape')(class_logits)
    class_scores = layers.Softmax(axis=-1, name='cls_softmax')(class_logits)

    # 中心点偏移量
    predict_deltas = layers.Conv2D(2 * num_anchors, kernel_size=(1, 1), name='deltas')(fc_output)
    predict_deltas = layers.Reshape(target_shape=(-1, 2), name='deltas_reshape')(predict_deltas)

    # 侧边精调(只需要预测x偏移即可)
    predict_side_deltas = layers.Conv2D(num_anchors, kernel_size=(1, 1), name='side_deltas')(fc_output)
    predict_side_deltas = layers.Reshape(target_shape=(-1, 1), name='side_deltas_reshape')(
        predict_side_deltas)
        
    model = Model(inputs=[input_image], outputs=[class_scores, predict_deltas, predict_side_deltas])
    return model


