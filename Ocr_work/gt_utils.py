import numpy as np


def linear_fit_y(xs, ys, x_list):
    #---------------------------------------#
    #   线性函数拟合两点(x1,y1),(x2,y2)；并求得x_list在的取值
    #   xs:  [x1,x2]
    #   ys:  [y1,y2]
    #   x_list: x轴坐标点,numpy数组 [n]
    #---------------------------------------#
    if xs[0] == xs[1]:  # 垂直线
        return np.ones_like(x_list) * np.mean(ys)
    elif ys[0] == ys[1]:  # 水平线
        return np.ones_like(x_list) * ys[0]
    else:
        fn = np.poly1d(np.polyfit(xs, ys, 1))  # 一元线性函数
        return fn(x_list)


def get_min_max_y(quadrilateral, xs):
    #---------------------------------------#
    #   获取指定x值坐标点集合四边形上的y轴最小值和最大值
    #   quadrilateral: 四边形坐标；x1,y1,x2,y2,x3,y3,x4,y4
    #   xs: x轴坐标点,numpy数组 [n]
    #---------------------------------------#
    x1, y1, x2, y2, x3, y3, x4, y4 = quadrilateral.tolist()
    y_val_1 = linear_fit_y(np.array([x1, x2]), np.array([y1, y2]), xs)
    y_val_2 = linear_fit_y(np.array([x2, x3]), np.array([y2, y3]), xs)
    y_val_3 = linear_fit_y(np.array([x3, x4]), np.array([y3, y4]), xs)
    y_val_4 = linear_fit_y(np.array([x4, x1]), np.array([y4, y1]), xs)
    y_val_min = []
    y_val_max = []
    for i in range(len(xs)):
        y_val = []
        if min(x1, x2) <= xs[i] <= max(x1, x2):
            y_val.append(y_val_1[i])
        if min(x2, x3) <= xs[i] <= max(x2, x3):
            y_val.append(y_val_2[i])
        if min(x3, x4) <= xs[i] <= max(x3, x4):
            y_val.append(y_val_3[i])
        if min(x4, x1) <= xs[i] <= max(x4, x1):
            y_val.append(y_val_4[i])
        # print("y_val:{}".format(y_val))
        y_val_min.append(min(y_val))
        y_val_max.append(max(y_val))

    return np.array(y_val_min), np.array(y_val_max)


def get_xs_in_range(x_array, x_min, x_max):
    #---------------------------------------#
    #   获取分割坐标点
    #   x_array: 宽度方向分割坐标点数组；0~image_width,间隔16 ；如:[0,16,32,...608]
    #   x_min: 四边形x最小值
    #   x_max: 四边形x最大值
    #---------------------------------------#
    indices = np.logical_and(x_array >= x_min, x_array <= x_max)
    xs = x_array[indices]
    # 处理两端的值
    if xs.shape[0] == 0 or xs[0] > x_min:
        xs = np.insert(xs, 0, x_min)
    if xs.shape[0] == 0 or xs[-1] < x_max:
        xs = np.append(xs, x_max)
    return xs


def gen_gt_from_quadrilaterals(gt_quadrilaterals, image_shape, width_stride):
    h, w = list(image_shape)[:2]
    x_array = np.arange(0, w + 1, width_stride, np.float32)  # 固定宽度间隔的x坐标点
    # 每个四边形x 最小值和最大值
    x_min_np = np.min(gt_quadrilaterals[:, ::2], axis=1)
    x_max_np = np.max(gt_quadrilaterals[:, ::2], axis=1)
    gt_boxes = []
    gt_class_ids = []
    for i in np.arange(len(gt_quadrilaterals)):
        xs = get_xs_in_range(x_array, x_min_np[i], x_max_np[i])  # 获取四边形内的x中坐标点
        ys_min, ys_max = get_min_max_y(gt_quadrilaterals[i], xs)
        # 为每个四边形生成固定宽度的gt
        for j in range(len(xs) - 1):
            x1, x2 = xs[j], xs[j + 1]
            y1, y2 = np.min(ys_min[j:j + 2]), np.max(ys_max[j:j + 2])
            gt_boxes.append([x1, y1, x2, y2])
            gt_class_ids.append(1)
    gt_boxes = np.reshape(np.array(gt_boxes), (-1, 4))
    gt_class_ids = np.reshape(np.array(gt_class_ids), (-1,))
    # 过滤高度太小的边框
    height = gt_boxes[:, 3] - gt_boxes[:, 1]
    width = gt_boxes[:, 2] - gt_boxes[:, 0]
    indices = np.where(np.logical_and(height >= 8, width >= 2))
    return gt_boxes[indices], gt_class_ids[indices]
