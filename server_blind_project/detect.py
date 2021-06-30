import base64
import colorsys
import datetime
import json
import os
import re
import signal
import threading
import time
import wave
from collections import Counter
from io import BytesIO

import cv2
import numpy as np
# import pyaudio
import requests
from aip import AipOcr, AipSpeech
from PIL import Image, ImageDraw, ImageFont
from pydub import AudioSegment

mode_classes = [
    [],
    ['person','cup','apple','chair','diningtable','book','banana','laptop',"cell phone"],
    ['person','car','bicycle','motorbike','bus','truck','traffic light','chair','diningtable',"traffic sign"],
    ['book'],
    ['person','car','bicycle','motorbike','bus','truck','chair','diningtable',],
]

""" 你的 APPID AK SK """
# APP_ID = "17191980"
# API_KEY = "ewWcl6GQ4Kh7qF2ys8sFvfgI"
# SECRET_KEY = "OO5g9xOxq96ZHKQzlvWyb2Uog6ePqu2U"
APP_ID = "22729954"
API_KEY = "LrVGobWoa29Q1lkimusywdwA"
SECRET_KEY = "wacYgjZG8mPMTGAGuT7oqOCv0zinkmux"
client_for_speak = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

# ----------------------------#
#   读取声音文件
# ----------------------------#
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

# ----------------------------#
#   用于检测判断与播放声音
# ----------------------------#
class Detect():     ##属性+方法
    def __init__(self):
        self.stop_signal = False
        self.speck_speed = 15
        #这里没有调用下面的函数，函数的功能如何实现？？

    def play_wav(self, dir, name):
        wf = wave.open(dir + "/" + name + ".wav", 'rb')
        chunk = 1024
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True)
        data = wf.readframes(chunk)
        while len(data) > 0:
            if self.stop_signal:
                break
            stream.write(data)
            data = wf.readframes(chunk)

        stream.close()
        p.terminate()

    def mp3towav(self, dir, name):
        print(dir + "/" + name + ".mp3")
        mp3 = AudioSegment.from_file(dir + "/" + name + ".mp3", format="mp3")
        # 修改对象参数
        wav = mp3.set_frame_rate(16000)
        wav = wav.set_channels(1)
        wav = wav.set_sample_width(2)
        # 导出wav文件
        wav.export(dir + "/" + name + ".wav", format='wav', )

    def play_sentence(self, sentence):
        # ----------------------------#
        #   转为mp3
        # ----------------------------#
        result = client_for_speak.synthesis(sentence, 'zh', 2, {
            'vol': 5, 'per': 0, 'spd': self.speck_speed
        })

            if not isinstance(result, dict):        #函数isinstance()可以判断一个变量的类型，既可以用在Python内置的数据类型如str、list、dict，也可以用在我们自定义的类，它们本质上都是数据类型。
            with open('mp3andwav/sentence.mp3', 'wb') as f:
                f.write(result)
        else:
            time.sleep(1)   #等待1
            return
        # ----------------------------#
        #   mp3转wav
        # ----------------------------#
        self.mp3towav("mp3andwav", "sentence")

        # ----------------------------#
        #   播放
        # ----------------------------#
        self.play_wav("mp3andwav", "sentence")

        # ----------------------------#
        #   休眠
        # ----------------------------#
        time.sleep(1)

    def save_images(self, image, path, file, bounding_boxes_all, class_all, traffic_boxes, traffic_class, face_bounding_boxes, face_class_all, mode, face_mode):
        '''
        mode的值可选范围为：0,1,2,3,4,5,6,7
        0为无工作状态
        1为室内模式
        2为室外模式
        3为阅读模式
        4为障碍物识别模式
        5为交通灯识别模式
        6为交通标志牌识别模式
        7为盲道和斑马线识别
        face_mode的值可选范围为：0,1,2,3
        0为无工作状态
        1为正常模式
        2为仅人脸识别
        3为表情识别
        '''
        image = Image.fromarray(np.uint8(image))
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = 5

        nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')

        classes = []
        bounding_boxes = []
        if mode == 1:       #1为室内模式
            try:
                for i,j in zip(class_all, bounding_boxes_all):  #zip 返回以元组为元素的列表
                    if i in mode_classes[mode]:
                        classes.append(i)
                        bounding_boxes.append(j)
            except:
                pass

        elif mode == 2:     #2为室外模式
            try:
                for i,j in zip(class_all, bounding_boxes_all):
                    if i in mode_classes[mode]:
                        classes.append(i)
                        bounding_boxes.append(j)
                for i,j in zip(traffic_class, traffic_boxes):
                    classes.append(i)
                    bounding_boxes.append(j)
            except:
                pass

        elif mode == 4 or mode == 7 or mode == 8:
            try:
                for i,j in zip(class_all,bounding_boxes_all):
                    if i in mode_classes[mode]:
                        classes.append(i)
                        bounding_boxes.append(j)
            except:
                pass


        elif mode == 5:
            try:
                for i,j in zip(traffic_class,traffic_boxes):
                    classes.append(i)
                    bounding_boxes.append(j)
            except:
                pass

        elif mode == 6:
            try:
                for i,j in zip(traffic_class,traffic_boxes):
                    classes.append(i)
                    bounding_boxes.append(j)
            except:
                pass
                
        if face_mode == 1:
            try:
                for i,j in zip(face_class_all,face_bounding_boxes[:, :4]):
                    classes.append(i)
                    bounding_boxes.append(j)
            except:
                pass
        elif face_mode == 2:
            try:
                for i,j in zip(face_class_all,face_bounding_boxes[:, :4]):
                    i = i.split(";")[0]
                    classes.append(i)
                    bounding_boxes.append(j)
            except:
                pass
        
        elif face_mode == 3:
            try:
                for i,j in zip(face_class_all,face_bounding_boxes[:, :4]):
                    i = i.split(";")[-1]
                    classes.append(i)
                    bounding_boxes.append(j)
            except:
                pass
                
        if (mode != 0 or face_mode != 0):
            classes_sentence = ""
            for cls in classes:
                classes_sentence += cls + " ; "
            file.write(nowTime + " : " + classes_sentence + "\n")

            for i, c in enumerate(classes):
                predicted_class = c

                left, top, right, bottom = bounding_boxes[i]
                top = top - 5
                left = left - 5
                bottom = bottom + 5
                right = right + 5

                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
                right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

                # 画框框
                label = '{}'.format(predicted_class)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=(255,0,0))

                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=(255,0,0))
                draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
                del draw

            image.save(path + nowTime + ".jpg")
