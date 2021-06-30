class Config(object):
    BATCH_SIZE = 2

    NUM_CLASSES = 1 + 1 

    VARIANCE = [0.1,0.2,0.1]

    # 训练样本
    ANCHORS_HEIGHT = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    ANCHORS_WIDTH = 16

    TRAIN_ANCHORS_PER_IMAGE = 128
    ANCHOR_POSITIVE_RATIO = 0.5

    RNN_UNITS = 64
    FC_UNITS = 256
    # 步长
    NET_STRIDE = 16

    # text proposal输出
    TEXT_PROPOSALS_WIDTH = 16

    # text line boxes超参数
    LINE_MIN_SCORE = 0.7
    MAX_HORIZONTAL_GAP = 50
    TEXT_LINE_NMS_THRESH = 0.3
    MIN_NUM_PROPOSALS = 1
    MIN_RATIO = 1.2
    MIN_V_OVERLAPS = 0.7
    MIN_SIZE_SIM = 0.7

    # 是否使用侧边改善
    USE_SIDE_REFINE = True
