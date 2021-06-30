import os
import cv2
import glob
import copy
import keras
import random
import numpy as np
import tensorflow as tf
from random import shuffle
from PIL import Image,ImageDraw
from utils.gt_utils import gen_gt_from_quadrilaterals
from keras.objectives import categorical_crossentropy
from keras.applications.imagenet_utils import preprocess_input
def _softmax_loss(y_true, y_pred):
    y_pred = tf.maximum(y_pred, 1e-7)
    softmax_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred),
                                    axis=-1)
    return softmax_loss
def cls_loss():
    def _cls_loss(y_true, y_pred):
        # y_true [batch_size, num_anchor, num_classes]
        # y_pred [batch_size, num_anchor, num_classes]
        labels         = y_true
        anchor_state   = y_true[:,:,-1] # -1 是需要忽略的, 0 是背景, 1 是存在目标
        classification = y_pred

        
        # 找出存在目标的先验框
        indices_for_object        = tf.where(keras.backend.equal(anchor_state, 1))
        labels_for_object         = tf.gather_nd(labels, indices_for_object)
        classification_for_object = tf.gather_nd(classification, indices_for_object)

        cls_loss_for_object = _softmax_loss(labels_for_object, classification_for_object)

        # 找出实际上为背景的先验框
        indices_for_back        = tf.where(keras.backend.equal(anchor_state, 0))
        labels_for_back         = tf.gather_nd(labels, indices_for_back)
        classification_for_back = tf.gather_nd(classification, indices_for_back)

        # 计算每一个先验框应该有的权重
        cls_loss_for_back = _softmax_loss(labels_for_back, classification_for_back)
        # cls_loss_for_object = tf.Print(cls_loss_for_object,[tf.shape(cls_loss_for_object),tf.shape(cls_loss_for_back),classification_for_object,labels_for_object])

        cls_loss = tf.concat([cls_loss_for_back,cls_loss_for_object],axis=0)
        # 找到正样本
        indices    = tf.where(keras.backend.not_equal(anchor_state, -1))
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        loss = keras.backend.sum(cls_loss) / normalizer

        # # # 将所获得的loss除上正样本的数量
        # cls_loss_for_object = keras.backend.mean(cls_loss_for_object)
        # cls_loss_for_back = keras.backend.mean(cls_loss_for_back)
        # # 总的loss
        # loss = cls_loss_for_object + cls_loss_for_back
        return loss
    return _cls_loss

def smooth_l1(sigma=3.0):
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        # 找到正样本
        indices           = tf.where(keras.backend.equal(anchor_state, 1))
        regression        = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # 计算 smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )
    
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        loss = keras.backend.sum(regression_loss) / normalizer

        return loss

    return _smooth_l1

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(image_annotation, input_shape, max_quadrilaterals=100, jitter=0, hue=.1, sat=1.2, val=1.2):
    '''random preprocessing for real-time data augmentation'''
    image = Image.open(image_annotation['image_path'])
    iw, ih = image.size
    h, w = input_shape
    quadrilaterals = copy.deepcopy(image_annotation['quadrilaterals'])
    # 对图像进行缩放并且进行长和宽的扭曲
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(1,1.5)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # 将图像多余的部分加上灰条
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # 翻转图像
    flip = 1
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 色域扭曲
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
    x[..., 0] += hue*360
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x[:,:, 0]>360, 0] = 360
    x[:, :, 1:][x[:, :, 1:]>1] = 1
    x[x<0] = 0
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255 # numpy array, 0 to 1

    # 将quadrilaterals进行调整
    if len(quadrilaterals)>0:
        np.random.shuffle(quadrilaterals)
        quadrilaterals[:, ::2] = quadrilaterals[:, ::2]*nw/iw + dx
        quadrilaterals[:, 1::2] = quadrilaterals[:, 1::2]*nh/ih + dy
        if flip: 
            quadrilaterals[:, ::2] = w - quadrilaterals[:, ::2]
            lt_x, lt_y, rt_x, rt_y, rb_x, rb_y, lb_x, lb_y = np.split(quadrilaterals, 8, axis=1)
            quadrilaterals = np.concatenate([rt_x, rt_y, lt_x, lt_y, lb_x, lb_y, rb_x, rb_y], axis=1)

        quadrilaterals[quadrilaterals<0] = 0
        quadrilaterals[:, ::2][quadrilaterals[:, ::2]>w] = w
        quadrilaterals[:, 1::2][quadrilaterals[:, 1::2]>h] = h
        if len(quadrilaterals)>max_quadrilaterals: quadrilaterals = quadrilaterals[:max_quadrilaterals]
    return image_data, quadrilaterals

class Generator(object):
    def __init__(self, file_paths, image_dir, img_size, batch_size, bbox_util, stride = 16):
        self.img_size = img_size

        self.file_paths = file_paths
        self.image_annotations = [self.load_annotation(path, image_dir) for path in self.file_paths]
        # 过滤不存在的图像，ICDAR2017中部分图像找不到
        self.image_annotations = [ann for ann in self.image_annotations if os.path.exists(ann['image_path'])]

        self.batch_size = batch_size
        self.stride = stride
        self.bbox_util = bbox_util


    def load_annotation(self, annotation_path, image_dir):
        image_annotation = {}
        # 返回最后的文件名
        base_name = os.path.basename(annotation_path)
        image_name = base_name[3:-3] + '*'
        image_annotation["annotation_path"] = annotation_path
        # 根据图片名进行匹配
        image_annotation["image_path"] = glob.glob(os.path.join(image_dir, image_name))[0]

        # 读取边框标注
        bbox = []
        quadrilateral = []  # 四边形

        with open(annotation_path, "r", encoding='utf-8') as f:
            lines = f.read().encode('utf-8').decode('utf-8-sig').splitlines()

        for line in lines:
            line = line.strip().split(",")
            # 左上、右上、右下、左下 四个坐标 如：377,117,463,117,465,130,378,130
            lt_x, lt_y, rt_x, rt_y, rb_x, rb_y, lb_x, lb_y = map(float, line[:8])
            # 获取平行四边形
            quadrilateral.append([lt_x, lt_y, rt_x, rt_y, rb_x, rb_y, lb_x, lb_y])

        # 将所有的框放在一起
        image_annotation["quadrilaterals"] = np.asarray(quadrilateral, np.float32).reshape((-1, 8))
        return image_annotation

    def get_len(self):
        return len(self.image_annotations)
    
    def generate(self):
        while True:
            shuffle(self.image_annotations)
            batch_images = []
            batch_class_scores = []
            batch_deltas = []
            batch_side_deltas = []
            for i, image_annotation in enumerate(self.image_annotations): 

                image_data, quadrilaterals = get_random_data(image_annotation,self.img_size) 
                gt_boxes, class_ids = gen_gt_from_quadrilaterals(quadrilaterals,self.img_size,self.stride)
                
                # image = Image.fromarray(np.uint8(image_data))
                # for box in gt_boxes:
                #     thickness = 3
                #     # 获取长方形
                #     left, top, right, bottom = box
                #     print(box)
                #     draw = ImageDraw.Draw(image)
                #     for k in range(thickness):
                #         draw.rectangle([left + k, top + k, right - k, bottom - k],outline=(255,255,255))
                #     print(self.stride)
                # image.show()

                # if len(gt_boxes)!=0:
                #     gt_boxes = np.array(gt_boxes,dtype=np.float32)
                #     gt_boxes[:,0] = gt_boxes[:,0]/self.img_size[1]
                #     gt_boxes[:,1] = gt_boxes[:,1]/self.img_size[0]
                #     gt_boxes[:,2] = gt_boxes[:,2]/self.img_size[1]
                #     gt_boxes[:,3] = gt_boxes[:,3]/self.img_size[0]
                    
                assignment = self.bbox_util.assign_boxes(gt_boxes)

                num_regions = 128
                
                classification = assignment[:,-2:]
                regression = assignment[:,[0,1,2,-1]]
                
                box_mask = classification[:,-1]
                mask_pos = box_mask>0
                num_pos = len(box_mask[mask_pos])
                if num_pos > num_regions/2:
                    val_locs = random.sample(range(num_pos), int(num_pos - num_regions/2))
                    temp_classification = classification[mask_pos]
                    temp_regression = regression[mask_pos]
                    temp_classification[val_locs,-1] = -1
                    temp_regression[val_locs,-1] = -1
                    classification[mask_pos] = temp_classification
                    regression[mask_pos] = temp_regression
                    
                box_mask = classification[:,-1]
                mask_neg = box_mask==0
                num_neg = len(box_mask[mask_neg])
                mask_pos = box_mask>0
                num_pos = len(box_mask[mask_pos])
                if len(box_mask[mask_neg]) + num_pos > num_regions or len(box_mask[mask_neg]) > num_regions/2:
                    reduce_num  = max(int(num_neg + num_pos - num_regions), int(num_neg - num_regions/2))
                    val_locs = random.sample(range(num_neg), reduce_num)
                    temp_classification = classification[mask_neg]
                    temp_classification[val_locs,-1] = -1
                    classification[mask_neg] = temp_classification
                    
                classification = np.reshape(classification,[-1,2])
                regression = np.reshape(regression,[-1,4])

                batch_images.append(image_data)
                batch_class_scores.append(classification)
                batch_deltas.append(regression[:,[0,1,-1]])
                batch_side_deltas.append(regression[:,[2,-1]])
                
                if len(batch_images) == self.batch_size:
                    batch_images = np.array(batch_images)
                    batch_class_scores = np.array(batch_class_scores)
                    batch_deltas = np.array(batch_deltas)
                    batch_side_deltas = np.array(batch_side_deltas)
                    
                    yield preprocess_input(batch_images), [batch_class_scores, batch_deltas, batch_side_deltas]
                    batch_images = []
                    batch_class_scores = []
                    batch_deltas = []
                    batch_side_deltas = []