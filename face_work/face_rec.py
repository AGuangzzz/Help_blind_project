import os
import re

import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image

from .emotion_nets.mobilenet import MobileNet
from .facenet_nets.arcface import arcface
from .retinaface import Retinaface
from .utils.utils import Alignment_1,face_distance,compare_faces


def generate_faces(face_img, img_size=96):
    """
    将探测到的人脸进行增广
    :param face_img: 灰度化的单个人脸图
    :param img_size: 目标图片大小
    :return:
    """
    face_img = np.array(letterbox_image(Image.fromarray(np.uint8(face_img)), (img_size, img_size)))
    resized_images = list()
    resized_images.append(face_img[:, :])
    resized_images.append(cv2.flip(face_img[:, :], 1))
    resized_images.append(face_img[0:91, 0:91])
    resized_images.append(face_img[0:91, 4:95])
    resized_images.append(face_img[4:95, 0:91])
    resized_images.append(face_img[4:95, 4:95])

    for i in range(len(resized_images)):
        resized_images[i] = np.array(letterbox_image(Image.fromarray(np.uint8(resized_images[i])), (img_size, img_size)))
        resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
        resized_images[i] = resized_images[i] / 255.
    resized_images = np.array(resized_images)
    return resized_images

#----------------------------------------#
#   预处理训练图片
#----------------------------------------#
def letterbox_image(image, size):
    image = image.convert("RGB")
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    new_image = new_image.convert("L")

    return new_image



class face_rec():
    def __init__(self):
        self.graph = tf.get_default_graph()
        # 人脸定位部分！
        self.retinaface = Retinaface()

        # 读取数据库
        self.known_face_encodings = np.load("./model_data/known_face_encodings.npy")
        self.known_face_names = np.load("./model_data/known_face_names.npy")

        self.face_input_shape = [112,112,3]
        self.facenet = arcface(self.face_input_shape)
        self.facenet.load_weights("./model_data/arcface_weights.h5")

        # 表情检测部分！
        # 读取模型
        input_shape = [96,96,1]
        emotion_model_path = './model_data/emotion_weights.h5'
        self.emotion_labels = {0: '无表情', 1: '高兴', 2: '惊讶', 3: '沮丧', 4: '生气', \
                               5: '恶心', 6: '害怕', 7:'轻蔑'}
        self.emotion_classifier = MobileNet(input_shape=input_shape,classes=len(self.emotion_labels))
        self.emotion_classifier.load_weights(emotion_model_path)
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]

    def add_face(self, image, name):
        with self.graph.as_default():
            reg = "[，。、？]"
            name = re.sub(reg,"",name)
            
            height, width, _ = np.shape(image)
            # 彩色图像
            rgb_small_frame = image
            # 根据上述参数进行人脸检测
            face_locations = self.retinaface.detect_image(rgb_small_frame)

            # 更变格式，用于传入face_recognition识别
            ids = []
            class_all = []
            bounding_boxes = []

            if len(face_locations) == 0 or len(face_locations) >= 2 :
                return np.array(bounding_boxes), np.array(class_all), np.array(ids)

            face_locations = np.array(face_locations, dtype=np.int32)
            face_locations[:, [0,2]] = np.clip(face_locations[:, [0,2]], 0, width)
            face_locations[:, [1,3]] = np.clip(face_locations[:, [1,3]], 0, height)

            best_face_location = np.array(face_locations)[0]

            # 截取图像
            crop_img = rgb_small_frame[int(best_face_location[1]):int(best_face_location[3]), int(best_face_location[0]):int(best_face_location[2])]
            
            landmark = np.reshape(best_face_location[5:],(5,2)) - np.array([int(best_face_location[0]),int(best_face_location[1])])

            crop_img,_ = Alignment_1(crop_img,landmark)
            crop_img = np.array(self.letterbox_image(Image.fromarray(np.uint8(crop_img)),(self.face_input_shape[0],self.face_input_shape[1])))/255
            crop_img = np.expand_dims(crop_img,0)
            # 利用facenet_model计算128维特征向量
            face_encodings = self.facenet.predict(crop_img)

            self.known_face_encodings = [face_encoding for face_encoding in self.known_face_encodings]
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names = [known_face_name for known_face_name in self.known_face_names]
            self.known_face_names.append(name)

            np.save("model_data/known_face_encodings.npy",self.known_face_encodings)
            np.save("model_data/known_face_names.npy",self.known_face_names)
            if os.path.exists("face_dataset/"+name):
                Image.fromarray(np.uint8(image)).save("face_dataset/"+name+"/"+name+".jpg")
            else:
                os.mkdir("face_dataset/"+name)
                Image.fromarray(np.uint8(image)).save("face_dataset/"+name+"/"+name+".jpg")

            return np.array(face_locations), np.array(class_all), np.array(ids)

    def letterbox_image(self, image, size):
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (255,255,255))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image

    def recognize(self, image):
        with self.graph.as_default():
            height, width, _ = np.shape(image)
            # 彩色图像
            rgb_small_frame = image
            # 获取灰度图像
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # 灰度图处理
            gray_image = np.squeeze(gray)
            gray_image = gray_image.astype('uint8')

            # 根据上述参数进行人脸检测
            face_locations = self.retinaface.detect_image(rgb_small_frame)

            # 更变格式，用于传入face_recognition识别
            ids = []
            class_all = []
            bounding_boxes = []
            if len(face_locations) == 0:
                return np.array(bounding_boxes), np.array(class_all), np.array(ids)
                
            #-----------------------------------------------#
            #   对检测到的人脸进行编码
            #-----------------------------------------------#
            face_encodings = []
            for result in face_locations:
                # 截取图像
                crop_img = np.array(rgb_small_frame)[int(result[1]):int(result[3]), int(result[0]):int(result[2])]
                landmark = np.reshape(result[5:],(5,2)) - np.array([int(result[0]),int(result[1])])
                crop_img,_ = Alignment_1(crop_img,landmark)

                crop_img = np.array(self.letterbox_image(Image.fromarray(np.uint8(crop_img)),(self.face_input_shape[1],self.face_input_shape[0])))/255
                # cv2.imshow("123",crop_img)
                # cv2.waitKey(0)
                
                crop_img = np.expand_dims(crop_img,0)
                # 利用facenet_model计算128维特征向量
                face_encoding = self.facenet.predict(crop_img)[0]
                face_encodings.append(face_encoding)

            # 用于存储表情
            emotion_text = []
            face_names = []

            for face_encoding, b in zip(face_encodings, face_locations):
                # 人脸识别
                matches = compare_faces(self.known_face_encodings, face_encoding, tolerance=1)
                name = "未知"
                face_distances = face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                face_names.append(name)

                if b[0] < 0 or b[1] < 0:
                    continue

                top = max(0, np.floor(int(b[1]) - 10 + 0.5).astype('int32'))
                left = max(0, np.floor(int(b[0]) - 10 + 0.5).astype('int32'))
                bottom = min(np.shape(image)[0], np.floor(int(b[3]) + 10 + 0.5).astype('int32'))
                right = min(np.shape(image)[1], np.floor(int(b[2]) + 10 + 0.5).astype('int32'))

                face_part = gray_image[top:bottom, left:right]
                faces_img_gray = generate_faces(face_part)
                
                # 预测结果线性加权
                results = self.emotion_classifier.predict(faces_img_gray)
                result_mean = np.mean(results, axis=0).reshape(-1)
                
                emotion = ""
                top_k_idx = result_mean.argsort()[::-1][0:2]
                probabilitys = []
                for i, index in enumerate(top_k_idx):
                    probabilitys.append(result_mean[index])
                    emotion += self.emotion_labels[index]
                    if i != 1:
                        emotion += "或"
                emotion_text.append(emotion)

            i = 0
            for face_name, emotion in zip(face_names, emotion_text):
                i = i + 1
                ids.append(i)
                class_all.append(face_name + ";" + emotion)
            return np.array(face_locations), np.array(class_all), np.array(ids)
