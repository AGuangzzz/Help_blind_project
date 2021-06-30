from .nets.pspnet import pspnet
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import colorsys
import copy
import os
import cv2
import tensorflow as tf
#--------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、backbone和num_classes
#   都需要修改！
#--------------------------------------------#
class Pspnet(object):
    _defaults = {
        "model_path"        : 'model_data/pspnet_blind.h5',
        "backbone"          : "mobilenet",
        "model_image_size"  : (473, 473, 3),
        "num_classes"       : 3,
        "downsample_factor" : 16,
        "blend"             : True,
    }

    #---------------------------------------------------#
    #   初始化PSPNET
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.graph = tf.get_default_graph()
        self.generate()

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):##生成框，对象和Index
        self.model = pspnet(self.num_classes,self.model_image_size,
                    downsample_factor=self.downsample_factor, backbone=self.backbone, aux_branch=False)

        self.model.load_weights(self.model_path)
        print('{} model loaded.'.format(self.model_path))
        
        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                    (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                    (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 12)]
        else:
            # 画框设置不同的颜色
            hsv_tuples = [(x / len(self.class_names), 1., 1.)
                        for x in range(len(self.class_names))]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors))

    def letterbox_image(self ,image, size):
        '''resize image with unchanged aspect ratio using padding'''
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))   #Creates a new image with the given mode and size.
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))      #Pastes another image into this image. 
        return new_image,nw,nh

    def extract_bboxes(self, mask):
        # 利用语义分割的mask找到包围它的框
        boxes = np.zeros([self.num_classes-1, 4], dtype=np.int32)
        for i in range(1, self.num_classes):
            m = (np.array(mask) == i)
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                x2 += 1
                y2 += 1
            else:
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[i-1] = np.array([x1, y1, x2, y2])
        return boxes.astype(np.int32)

    def extract_centers(self, mask):
        # 利用语义分割的mask找到包围它的框
        boxes = np.zeros([self.num_classes-1, 2], dtype=np.int32)
        for i in range(1, self.num_classes):
            m = np.argwhere(np.array(mask) == i)
            if m.shape[0]:
                boxes[i-1, 1] = np.mean(m[:,0])
                boxes[i-1, 0] = np.mean(m[:,1])
            else:
                boxes[i-1, 1] = 0
                boxes[i-1, 0] = 0

        return boxes.astype(np.int32)

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        with self.graph.as_default():
            img = Image.fromarray(np.uint8(image))
            orininal_h = np.array(img).shape[0]
            orininal_w = np.array(img).shape[1]

            img, nw, nh = self.letterbox_image(img,(self.model_image_size[1],self.model_image_size[0]))
            img = [np.array(img)/255]
            img = np.asarray(img)
            
            pr = self.model.predict(img)[0]
            # 取出每一个像素点的分类结果
            pr = pr.argmax(axis=-1).reshape([self.model_image_size[0],self.model_image_size[1]])

            for c in range(1, self.num_classes):
                m = (pr == c)
                num = np.sum(m)
                if num < self.model_image_size[0]*self.model_image_size[1]*0.005:
                    pr[m] = 0

            kernel = np.ones((15,15),np.uint8)
            pr = cv2.erode(np.uint8(pr),kernel)
            pr = cv2.dilate(pr,kernel)

            pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]

            pr = Image.fromarray(np.uint8(pr)).resize((orininal_w,orininal_h), Image.NEAREST)

            bboxes  = self.extract_bboxes(pr)  
            centers = self.extract_centers(pr)

            return np.array(bboxes), np.array(centers), np.array(["盲道", "斑马线"])