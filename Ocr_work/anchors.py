import keras
import numpy as np


def generate_base_anchors(heights, width):
    w = np.array([width] * len(heights))
    h = np.array(heights)
    return np.stack([-0.5 * w, -0.5 * h, 0.5 * w, 0.5 * h], axis=1)


def shift(shape, stride, base_anchors):
    H, W = shape[0], shape[1]
    ctr_x = (np.array(np.arange(W), np.float32) + 0.5 ) * stride
    ctr_y = (np.array(np.arange(H), np.float32) + 0.5 ) * stride

    ctr_x, ctr_y = np.meshgrid(ctr_x, ctr_y)

    # 打平为1维,得到所有锚点的坐标
    ctr_x = np.reshape(ctr_x, [-1])
    ctr_y = np.reshape(ctr_y, [-1])
    #  (H*W,1,4)
    shifts = np.expand_dims(np.stack([ctr_x, ctr_y, ctr_x, ctr_y], axis=1), axis=1)
    # (1,anchor_num,4)
    base_anchors = np.expand_dims(base_anchors, axis=0)

    # (H*W,anchor_num,4)
    anchors = shifts + base_anchors
    # 转为(H*W*anchor_num,4) 返回
    return np.reshape(anchors, [-1, 4])


def generate_anchors(heights, width, stride, features_shape):
    base_anchors = generate_base_anchors(heights, width)
    anchors = shift(features_shape, stride, base_anchors)
    # anchors[:,[0,2]] = anchors[:,[0,2]]/features_shape[1]/stride
    # anchors[:,[1,3]] = anchors[:,[1,3]]/features_shape[0]/stride
    return anchors



def main():
    base_anchors = generate_base_anchors([11, 16, 23, 33, 48, 68, 97, 139, 198, 283], 16)
    anchors = generate_anchors([11, 16, 23, 33, 48, 68, 97, 139, 198, 283], 16, 16,[38,38])
    print(anchors)
if __name__ == '__main__':
    main()
