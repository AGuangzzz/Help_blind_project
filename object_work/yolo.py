import os
import cv2
import copy
import colorsys
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from PIL import Image
from keras import backend as K
from keras.layers import Input
from .nets.yolo4 import yolo_body, yolo_eval
from .traffic_nets.mobilenet import MobileNet
from .utils.utils import letterbox_image

# ----------------------------#
#   红灯检测
# ----------------------------#
def detect_red(img, Threshold=0.01):
    desired_dim = np.shape(np.array(img))
    img = np.array(img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)      #RGB转换到HSV

    lower_red = np.array([0, 70, 70])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    lower_red = np.array([160, 70, 70])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    mask = mask0 + mask1

    rate = np.count_nonzero(mask) / (desired_dim[0] * desired_dim[1])

    if rate > Threshold:
        return True
    else:
        return False


# ----------------------------#
#   绿灯检测
# ----------------------------#
def detect_green(img, Threshold=0.01):
    desired_dim = np.shape(np.array(img))
    img = np.array(img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_red = np.array([45, 70, 70])
    upper_red = np.array([90, 255, 255])
    mask = cv2.inRange(img_hsv, lower_red, upper_red)

    rate = np.count_nonzero(mask) / (desired_dim[0] * desired_dim[1])

    if rate > Threshold:
        return True
    else:
        return False


# ----------------------------#
#   把所有交通灯记录
# ----------------------------#
def read_traffic_lights(image, boxes):      #boxes从哪里来的，四个值
    flag = "traffic light"
    ymin, xmin, ymax, xmax = tuple(boxes)
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    crop_img = image.crop((left, top, right, bottom))       #图片裁剪操作，四个参数
    if detect_green(crop_img):
        flag = "绿灯"
    if detect_red(crop_img):
        flag = "红灯"

    return flag


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo_blind.h5',
        "classes_path": 'model_data/yolo_blind.txt',
        "anchors_path": 'model_data/yolo_anchors.txt',
        
        "traffic_model_path": 'model_data/traffic_weights.h5',
        "traffic_classes_path": 'model_data/traffic_classes.txt',

        "light_model_path": 'model_data/light_weights.h5',
        "light_classes_path": 'model_data/light_classes.txt',
        
        "score": 0.3,
        "iou": 0.3,
        "model_image_size": (608, 608)
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化yolo
    # ---------------------------------------------------#
    def __init__(self, traffic=1, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(**kwargs)
        self.sess = KTF.get_session()
        self.graph = tf.get_default_graph()
        self.traffic = traffic
        self.anchors = self._get_anchors()
        self.class_names = self._get_class()
        self.boxes, self.scores, self.classes = self.generate()#这里已经获得框，类，分

        self.traffic_class_names = self._get_traffic_class() #   获得交通标志牌所有的分类
        self.traffic_model = self.load_traffic_model()#   获得交通标志牌分类网络
        self.light_class_names = self._get_light_class()#   获得交通信号灯所有的分类
        self.light_model = self.load_light_model() #   获得交通信号灯分类网络    

    # ---------------------------------------------------#
    #   获得交通标志牌所有的分类
    # ---------------------------------------------------#
    def _get_traffic_class(self):
        classes_path = os.path.expanduser(self.traffic_classes_path)
        with open(classes_path,encoding="utf-8") as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names                                                      

    # ---------------------------------------------------#
    #   获得交通标志牌分类网络
    # ---------------------------------------------------#
    def load_traffic_model(self):
        model = MobileNet(classes = len(self.traffic_class_names))
        model.load_weights(self.traffic_model_path)
        return model                                                      #返回训练好的mobilenet模型

    # ---------------------------------------------------#
    #   获得交通信号灯所有的分类
    # ---------------------------------------------------#
    def _get_light_class(self):
        classes_path = os.path.expanduser(self.light_classes_path)
        with open(classes_path,encoding="utf-8") as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   获得交通信号灯分类网络
    # ---------------------------------------------------#
    def load_light_model(self):
        model = MobileNet(classes = len(self.light_class_names))#引入网络
        model.load_weights(self.light_model_path)#加载训练好的参数权重
        return model

    # ---------------------------------------------------#
    #   获得目标检测网络的所有分类
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]      #strip去掉开头和结尾的空格
        return class_names                                  #得到所有分类names
    # ---------------------------------------------------#
    #   获得所有的先验框
    # ---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path,encoding="utf-8") as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]    
        return np.array(anchors).reshape(-1, 2)

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):     #返回boxes,score,classes
        model_path = os.path.expanduser(self.model_path)#model.path 对应的是yolo_blind.h5,与yolo-blind.txt区分开（.txt是所有类名称）
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # 计算anchor数量
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)#这里的class为yolo_blind.txt里的class

        # 载入模型。
        self.yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)

        self.yolo_model.load_weights(self.model_path)#model path 为yolo_blind显示的权重
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self.input_image_shape = K.placeholder(shape=(2,))

        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           num_classes, self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, img):
        with self.graph.as_default():
            image = Image.fromarray(np.uint8(img))
            # 调整图片使其符合输入要求
            new_image_size = self.model_image_size
            boxed_image = letterbox_image(image, new_image_size)
            image_data = np.array(boxed_image, dtype='float32')
            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

            # 预测结果
            out_boxes, out_scores, out_classes = self.sess.run(     #sess.run是干啥的
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })

            if self.traffic:    #已经定义过 traffic = 1
                small_pic = []
                for i, c in list(enumerate(out_classes)):##enumerate：枚举
                    predicted_class = self.class_names[c]
                    if predicted_class in ["traffic sign", "traffic light"]:
                        box = out_boxes[i]

                        top, left, bottom, right = copy.deepcopy(box)
                        top = max(0, np.floor(top + 0.5).astype('int32'))
                        left = max(0, np.floor(left + 0.5).astype('int32'))
                        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                        small_image = letterbox_image(image.crop((left, top, right, bottom)), (96, 96))
                        small_image = np.asarray(small_image)
                        small_image = small_image / 255
                        small_image = np.expand_dims(small_image, axis=0)

                        small_pic.append(small_image)

            # 用于存储表情
            bounding_boxes = []     #定义数组，存bounding_boxes
            class_all = []      ##定义数组，存class
            ids = []        ##定义数组，存id

            traffic_boxes = []
            traffic_class = []
            traffic_index = 0
            for i, c in list(enumerate(out_classes)):
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]
                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                if self.traffic:
                    if predicted_class in ["traffic sign"]:
                        traffic_model_Pre = self.traffic_model.predict(small_pic[traffic_index])
                        traffic_index = traffic_index + 1
                        traffic_predicted_class = self.traffic_class_names[np.argmax(traffic_model_Pre)]
                        # if np.max(traffic_model_Pre) < 0.5:
                        #     continue
                        # if np.max(traffic_model_Pre) * score < 0.3:
                        #     continue
                        traffic_class.append(traffic_predicted_class)
                        traffic_boxes.append([left, top, right, bottom])
                        continue

                    elif predicted_class in ["traffic light"]:
                        light_model_Pre = self.light_model.predict(small_pic[traffic_index])
                        traffic_index = traffic_index + 1
                        traffic_predicted_class = self.light_class_names[np.argmax(light_model_Pre)]
                        if traffic_predicted_class == "黄灯":
                            traffic_predicted_class = "红灯"
                            
                        if traffic_predicted_class in ["绿灯", "红灯"]:
                            traffic_class.append(traffic_predicted_class)
                            traffic_boxes.append([left, top, right, bottom])
                            continue

                    # if predicted_class in ["traffic light"]:
                    #     predicted_class = read_traffic_lights(image, box)
                    #     if predicted_class in ["绿灯", "红灯"]:
                    #         traffic_class.append(predicted_class)
                    #         traffic_boxes.append([left, top, right, bottom])
                    #         continue

                class_all.append(predicted_class)
                bounding_boxes.append([left, top, right, bottom])
                ids.append(i)

            return np.array(bounding_boxes), np.array(class_all), np.array(traffic_boxes), np.array(traffic_class), np.array(ids)

    def close_session(self):
        self.sess.close()
