import datetime
import json
import os
import threading
import time

#数字化编码的音频数据
import wave

from collections import Counter

import cv2
import numpy as np
import pyaudio
from draft import requests
from aip import AipOcr, AipSpeech
from PIL import Image, ImageDraw, ImageFont
from pydub import AudioSegment

from utils.utils import image_to_base64, image_to_base64_dep#少了一个函数


#对于整数的+-*/也是用浮点数方法计算，所以对于计算的四种错误，我们会设置不同的方式处理
# warn会提示警告，ignore不采取措施，raise显示错误，等等详情看文档
olderr = np.seterr(all='ignore')

#-----------------------------#
#   记录声音用的client
#-----------------------------#
#这里为啥要开3个应用
APP_ID = "22729954"
API_KEY = "LrVGobWoa29Q1lkimusywdwA"
SECRET_KEY = "wacYgjZG8mPMTGAGuT7oqOCv0zinkmux"
client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

#-----------------------------#
#   输出声音用的client
#-----------------------------#
APP_ID = "22808383"
API_KEY = "Cu8BfWlmVG70sOeyGaQlCkcV"
SECRET_KEY = "GhFN86sg3FpB0h7WVCSOprtgi2wqYAyx"
client_for_speak = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

#-----------------------------#
#   文字识别用的client
#-----------------------------#
APP_ID = "22729886"
API_KEY = "EDrxKsEyjZ6VZXPMc9yZrnlp"
SECRET_KEY = "Ax5HgTPasDqpLocaMIFQEjGWI6bQWkdH"
client_for_ocr = AipOcr(APP_ID, API_KEY, SECRET_KEY)

#-----------------------------#
#   不同模式所使用的类
#-----------------------------#
mode_classes = [
    [],
    ['person','cup','apple','chair','diningtable','book','banana','laptop',"cell phone"],
    ['person','car','bicycle','motorbike','bus','truck','traffic light','chair','diningtable',"traffic sign"],
    ['book'],
    ['person','car','bicycle','motorbike','bus','truck','chair','diningtable',],
    [],
    [],
    ['chair'],
    ['cup'],
]

# ----------------------------#
#   用于记录声音
# ----------------------------#
class Recorder():
    #多通道声音，1024位为一个数据块，传输速率为rate
    def __init__(self, chunk=1024, channels=1, rate=16000):
        self.CHUNK = chunk
        #pyaudio的一种规定音频格式
        self.FORMAT = pyaudio.paInt16#定义有符号整数16位输入格式
        self.CHANNELS = channels
        self.RATE = rate
        self._running = True
        self._frames = []

    #考虑多线程的重要性，防止阻塞和等待
    def start(self):
        #新建音频下载线程并且传递参数
        threading._start_new_thread(self.__recording, ())

#播放语音的函数
    def __recording(self):
        #控制系统运行
        self._running = True
        #接受传输帧
        self._frames = []
        #实例化一个音频播放系统
        p = pyaudio.PyAudio()
        #读入数据流
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        while (self._running):
            #chunk是数据流块
            data = stream.read(self.CHUNK)
            self._frames.append(data)
        stream.stop_stream()
        stream.close()
        p.terminate()

    def stop(self):
        self._running = False

    #设置音频数据的操作
    #存储函数
    def save(self, filename):
        p = pyaudio.PyAudio()
        if not filename.endswith(".wav"):#检查后缀文件
            filename = filename + ".wav"

        #音频文件都用二进制读写
        wf = wave.open(filename, 'wb')

        #声道数
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)

        '''
        Python3的字符串的编码语言用的是unicode编码，
        由于Python的字符串类型是str，在内存中以Unicode表示，一个字符对应若干字节，
        如果要在网络上传输，或保存在磁盘上就需要把str变成以字节为单位的bytes        
        '''
        #将字符串形式的帧 ：列表形式  全部存储到filename中
        wf.writeframes(b''.join(self._frames))
        wf.close()
        print("Saved")


#这个函数干嘛的，怪怪的
#计算摄像头-物体距离的式子:   均值
def inlinerDepth(object_photo):
    #reshape(-1) 转换成一列
    seq = object_photo.reshape(-1)
    #取出<4的元素===>数组
    seq = seq[(seq < 4)]
    #标准差和均值
    std = np.std(seq)
    mean = np.mean(seq)
    #
    inlier = seq[np.abs(seq - mean) < 1.2 *std]
    if len(inlier)==0:
        #中位数
        return np.median(object_photo)
    else:
        return np.mean(inlier)

# ----------------------------#
#   读取声音文件
# ----------------------------#

#读取声音文件要用二进制读取
#二进制读取全部内容
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

# ----------------------------#
#   发送图片的代码
# ----------------------------#
#request库函数

#这里向服务器写数据
def json_send(dataPModel,url):
    #请求头
    #请求报文可通过一个“Accept”报文头属性告诉服务端 客户端接受什么类型的响应。
    #如下报文头相当于告诉服务端，俺客户端能够接受的响应类型仅为纯文本数据啊，
    # 你丫别发其它什么图片啊，视频啊过来，那样我会歇菜的

    #客户端支持的访问服务器内容，客户端接收的格式和客户端的编码方式
    headers = {"Content-type": "application/json", "Accept": "text/plain", "charset": "UTF-8"}
    #dumps将json数据模型dict格式转换成字符串格式
    response = requests.post(url=url, headers=headers, data=json.dumps(dataPModel))
    response_text = response.text
    #loads再转换成dict模式
    return json.loads(response_text)


def run_send_sentence(sentence, sentence_url):
    try:
        dataPModel = {"sentence": sentence}
        # result = json_send(dataPModel, self.urls['sentence_url'])["result"]

        #只要result对应的body值
        result = json_send(dataPModel, sentence_url)["result"]
    except:
        pass

# ----------------------------#
#   用于检测判断与播放声音
# ----------------------------#
class Detect():
    def __init__(self, height, width, mode=0, face_mode=0, seg_mode=0, 
        navigation_mode=0, navigation_place=0, urls=None, speck_speed=8):
        # ----------------------------#
        #   目标信息
        # ----------------------------#
        self.object_bounding_boxes  = []
        self.object_distance        = []
        self.object_class_all       = []
        self.object_x_center        = []
        self.object_y_center        = []

        # ----------------------------#
        #   交通信息
        # ----------------------------#
        self.traffic_bounding_boxes = []
        self.traffic_class_all      = []

        # ----------------------------#
        #   文字识别结果
        # ----------------------------#
        self.result_list            = []
        self.orc_save_sentence      = ""

        # ----------------------------#
        #   人脸信息
        # ----------------------------#
        self.face_bounding_boxes    = []
        self.face_class_all         = []
        self.face_x_center          = []
        self.face_y_center          = []

        # ----------------------------#
        #   斑马线与盲道信息
        # ----------------------------#
        self.pspnet_bounding_boxes  = []
        self.pspnet_center          = []
        self.pspnet_class_all       = ["盲道", "斑马线"]

        # ----------------------------#
        #   导航信息
        # ----------------------------#
        self.navigation_place       = navigation_place
        self.navigation_end         = 0
        self.navigation_sentence    = ""

        # ----------------------------#
        #   输入宽高
        # ----------------------------#
        self.height                 = height
        self.width                  = width
        self.image                  = np.zeros([height,width,3])
        self.depth                  = np.zeros([height,width])

        # ----------------------------#
        #   各种模式
        # ----------------------------#
        self.mode                   = mode
        self.face_mode              = face_mode
        self.seg_mode               = seg_mode
        self.navigation_mode        = navigation_mode


        self.wake_signal            = False
        self.speak_flag             = True

        # ----------------------------#
        #   Flask用到的各种地址
        # ----------------------------#
        self.urls                   = urls

        # ----------------------------#
        #   语音速度
        # ----------------------------#
        self.speck_speed            = speck_speed

        #服务器端的数据
        self.object_list            = self._get_class("model_data/yolo_tiny_blind.txt")
        self.object_dic             = {
            "person": "人", "car": "汽车", "bicycle": "自行车", "motorbike": "摩托车", "bus": "公交车",
            "truck": "卡车", "traffic light": "交通信号灯", "cup": "杯子", "spoon": "勺子",
            "bowl": "碗", "apple": "苹果", "chair": "椅子", "diningtable": "桌子", "book": "书",
            "toothbrush": "牙刷", "banana": "香蕉", "laptop": "平板", "cell phone": "手机",
            "traffic sign": "交通标志牌"
        }

        #文件有这个路径吗？
        #列表相加，直接连接就可以
        self.traffic_list           = self._get_class("model_data/traffic_classes.txt") + ["无颜色", "绿灯", "红灯"]

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def _get_class(self, classes_path):
        #下面这个函数是用来展开缩写的路劲的，比如~，.等
        classes_path = os.path.expanduser(classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        #返回类别列表
        return class_names

    def recognize_init(self):
        # ----------------------------#
        #   目标信息
        # ----------------------------#
        self.object_bounding_boxes = []
        self.object_class_all = []
        self.object_x_center = []
        self.object_y_center = []
        self.object_distance = []

        # ----------------------------#
        #   交通信息
        # ----------------------------#
        self.traffic_bounding_boxes = []
        self.traffic_class_all = []

        # ----------------------------#
        #   文字识别结果
        # ----------------------------#
        self.result_list = []
        self.orc_save_sentence = ""

        # ----------------------------#
        #   人脸信息
        # ----------------------------#
        self.face_bounding_boxes = []
        self.face_class_all = []
        self.face_x_center = []
        self.face_y_center = []

        # ----------------------------#
        #   斑马线与盲道信息
        # ----------------------------#
        self.pspnet_bounding_boxes = []
        self.pspnet_center = []

    # ----------------------------------------#
    #   根据预测框的位置计算中心
    # ----------------------------------------#
    def computer_object_center(self):
        self.object_x_center = []
        self.object_y_center = []
        for bounding_box in self.object_bounding_boxes:
            center = (bounding_box[2] + bounding_box[0]) / 2
            self.object_x_center.append(center)
        for bounding_box in self.object_bounding_boxes:
            center = (bounding_box[3] + bounding_box[1]) / 2
            self.object_y_center.append(center)

    # ----------------------------------------#
    #   计算目标所处的方位
    # ----------------------------------------#
    def computer_pos(self, i):
        x_pos = self.object_x_center[i]
        y_pos = self.object_y_center[i]
        if x_pos < self.width * 0.3 and self.height * 0.7 >= y_pos >= self.height * 0.3:
            return "左边"
        elif x_pos < self.width * 0.3 and y_pos > self.height * 0.7:
            return "左下方"
        elif x_pos < self.width * 0.3 and y_pos < self.height * 0.3:
            return "左上方"
        elif self.width * 0.7 >= x_pos >= self.width * 0.3 and self.height * 0.7 > y_pos > self.height * 0.3:
            return "正前方"
        elif self.width * 0.7 >= x_pos >= self.width * 0.3 and y_pos > self.height * 0.7:
            return "正下方"
        elif self.width * 0.7 >= x_pos >= self.width * 0.3 and y_pos < self.height * 0.3:
            return "正上方"
        elif x_pos > self.width * 0.7 and self.height * 0.7 >= y_pos >= self.height * 0.3:
            return "右边"
        elif x_pos > self.width * 0.7 and y_pos > self.height * 0.7:
            return "右下方"
        elif x_pos > self.width * 0.7 and y_pos < self.height * 0.3:
            return "右上方"

    # ----------------------------------------#
    #   计算目标的距离
    # ----------------------------------------#
    def computer_object_distance(self, depth_image):
        for box in self.object_bounding_boxes:
            left, top, right, bottom = box
            height      = bottom-top
            width       = right-left
            crop_image  = depth_image[int(top+height*0.3):int(bottom-height*0.3),int(left+width*0.3):int(right-width*0.3)]
            
            distance    = inlinerDepth(crop_image)
            self.object_distance.append(distance)

    # ----------------------------------------#
    #   根据人脸框的位置计算中心
    # ----------------------------------------#
    def computer_face_center(self):
        self.face_x_center = []
        self.face_y_center = []
        for bounding_box in self.face_bounding_boxes:
            center = (bounding_box[2] + bounding_box[0]) / 2
            self.face_x_center.append(center)
        for bounding_box in self.face_bounding_boxes:
            center = (bounding_box[3] + bounding_box[1]) / 2
            self.face_y_center.append(center)

    # ----------------------------------------#
    #   计算人脸所处的方位
    # ----------------------------------------#
    def computer_face_pos(self, i):
        x_pos = self.face_x_center[i]
        y_pos = self.face_y_center[i]
        if x_pos < self.width * 0.3 and self.height * 0.7 >= y_pos >= self.height * 0.3:
            return "左边"
        elif x_pos < self.width * 0.3 and y_pos > self.height * 0.7:
            return "左下方"
        elif x_pos < self.width * 0.3 and y_pos < self.height * 0.3:
            return "左上方"
        elif self.width * 0.7 >= x_pos >= self.width * 0.3 and self.height * 0.7 > y_pos > self.height * 0.3:
            return "正前方"
        elif self.width * 0.7 >= x_pos >= self.width * 0.3 and y_pos > self.height * 0.7:
            return "正下方"
        elif self.width * 0.7 >= x_pos >= self.width * 0.3 and y_pos < self.height * 0.3:
            return "正上方"
        elif x_pos > self.width * 0.7 and self.height * 0.7 >= y_pos >= self.height * 0.3:
            return "右边"
        elif x_pos > self.width * 0.7 and y_pos > self.height * 0.7:
            return "右下方"
        elif x_pos > self.width * 0.7 and y_pos < self.height * 0.3:
            return "右上方"

    # ----------------------------------------#
    #   计算盲道所处的方位
    # ----------------------------------------#
    def computer_pspnet_pos(self, i):
        x_pos = self.pspnet_center[i][0]
        y_pos = self.pspnet_center[i][1]
        if x_pos < self.width * 0.3 and self.height * 0.7 >= y_pos >= self.height * 0.3:
            return "左边"
        elif x_pos < self.width * 0.3 and y_pos > self.height * 0.7:
            return "左下方"
        elif x_pos < self.width * 0.3 and y_pos < self.height * 0.3:
            return "左上方"
        elif self.width * 0.7 >= x_pos >= self.width * 0.3 and self.height * 0.7 > y_pos > self.height * 0.3:
            return "正前方"
        elif self.width * 0.7 >= x_pos >= self.width * 0.3 and y_pos > self.height * 0.7:
            return "正下方"
        elif self.width * 0.7 >= x_pos >= self.width * 0.3 and y_pos < self.height * 0.3:
            return "正上方"
        elif x_pos > self.width * 0.7 and self.height * 0.7 >= y_pos >= self.height * 0.3:
            return "右边"
        elif x_pos > self.width * 0.7 and y_pos > self.height * 0.7:
            return "右下方"
        elif x_pos > self.width * 0.7 and y_pos < self.height * 0.3:
            return "右上方"

    def recognize_all(self, image, depth):
        '''
        self.mode的值可选范围为：0,1,2,3,4,5,6,7,8
        0为无工作状态
        1为室内模式
        2为室外模式
        3为阅读模式
        4为障碍物识别模式
        5为交通灯识别模式
        6为交通标志牌识别模式
        7为椅子识别模式
        8为杯子识别模式
        
        self.face_mode的值可选范围为：0,1,2,3
        0为无工作状态
        1为正常模式
        2为仅人脸识别
        3为仅表情识别

        self.seg_mode的值可选范围为：0,1
        0为无工作状态
        1为开启斑马线识别
        '''
        self.image = image
        self.depth = depth
        self.recognize_init()

        #加密到base64编码
        #dataPmodel就是请求和响应的帧
        img_base64              = image_to_base64(image)
        #dataPmodel 就是对应一个字典，我们可以有不同的模式
        dataPModel              = {"img_base64": img_base64}
        dataPModel["mode"]      = self.mode
        dataPModel["face_mode"] = self.face_mode

        if self.wake_signal:
            return

        if self.mode == 1:
            result = json_send(dataPModel, self.urls['object_url'])['result']
            self.object_bounding_boxes, self.object_class_all, self.traffic_bounding_boxes, self.traffic_class_all = \
                result["bounding_boxes"], result["class_all"], result["traffic_boxes"], result["traffic_class"]
            self.computer_object_center()
            self.computer_object_distance(depth)

        if self.mode == 2:
            result = json_send(dataPModel, self.urls['object_url'])['result']
            self.object_bounding_boxes, self.object_class_all, self.traffic_bounding_boxes, self.traffic_class_all = \
                result["bounding_boxes"], result["class_all"], result["traffic_boxes"], result["traffic_class"]
            self.computer_object_center()
            self.computer_object_distance(depth)

        if self.mode == 3:
            result = json_send(dataPModel, self.urls['object_url'])['result']
            self.object_bounding_boxes, self.object_class_all, self.traffic_bounding_boxes, self.traffic_class_all = \
                result["bounding_boxes"], result["class_all"], result["traffic_boxes"], result["traffic_class"]
            self.computer_object_center()
            self.orc_save_sentence = result["orc_save_sentence"]

        if self.mode == 4 or self.mode == 7 or self.mode==8:
            result = json_send(dataPModel, self.urls['object_url'])['result']
            self.object_bounding_boxes, self.object_class_all, self.traffic_bounding_boxes, self.traffic_class_all = \
                result["bounding_boxes"], result["class_all"], result["traffic_boxes"], result["traffic_class"]
            self.computer_object_center()
            self.computer_object_distance(depth)
            
        if self.mode == 5 or self.mode == 6:
            result = json_send(dataPModel, self.urls['traffic_url'])['result']
            self.object_bounding_boxes, self.object_class_all, self.traffic_bounding_boxes, self.traffic_class_all = \
                result["bounding_boxes"], result["class_all"], result["traffic_boxes"], result["traffic_class"]
            self.computer_object_center()
            self.computer_object_distance(depth)
            
        if self.face_mode == 1 or self.face_mode == 2 or self.face_mode == 3:
            result = json_send(dataPModel,self.urls['face_url'])['result']
            self.face_bounding_boxes, self.face_class_all = result["face_locations"], result["class_all"]
            self.computer_face_center()
            
        if self.seg_mode == 1:
            result = json_send(dataPModel, self.urls['pspnet_url'])["result"]
            self.pspnet_bounding_boxes = result["bounding_boxes"]
            self.pspnet_center = result["centers"]

    def send_images(self, color_image, depth_map):
        if self.navigation_mode:
            img_base64      = image_to_base64(color_image)
            dep_img_base64  = image_to_base64_dep(depth_map*10)
            
            dataPModel      = {"img_base64": img_base64}
            dataPModel["dep_img_base64"]    = dep_img_base64
            dataPModel["navigation_mode"]   = self.navigation_mode
            dataPModel["navigation_place"]  = self.navigation_place
            result                          = json_send(dataPModel, self.urls['send_image_url'])['result']

            if result["navigation_end"] == 1:
                dataPModel                      = {"img_base64": img_base64}
                dataPModel["dep_img_base64"]    = dep_img_base64
                dataPModel["navigation_mode"]   = 0
                dataPModel["navigation_place"]  = 0
                result                          = json_send(dataPModel, self.urls['send_image_url'])['result']

            self.navigation_end                 = result["navigation_end"]
            self.navigation_sentence            = result["navigation_sentence"]

    def process_object(self):
        '''
        self.mode的值可选范围为：0,1,2,3,4,5,6,7,8
        0为无工作状态
        1为室内模式
        2为室外模式
        3为阅读模式
        4为障碍物识别模式
        5为交通灯识别模式
        6为交通标志牌识别模式
        7为椅子识别模式
        8为杯子识别模式
        '''
        # --------------------------#
        #   是否对用户提醒视角调整
        # --------------------------#
        adjust_flag = False
        
        sentence    = ""
        num_all     = np.zeros(len(self.object_list))
        pos_cls2pos = []
        pos_cls2dep = []
        # -------------------------------#
        #   创建全部的空列表
        # -------------------------------#
        for _ in num_all:
            pos_cls2pos.append([])
            pos_cls2dep.append([])
            
        # -------------------------------#
        #   判断每个类的数量以及位置
        # -------------------------------#
        for i, classe in enumerate(self.object_class_all):
            num_all[self.object_list.index(classe)] += 1
            pos_cls2pos[self.object_list.index(classe)].append(self.computer_pos(i))
            pos_cls2dep[self.object_list.index(classe)].append(self.object_distance[i])

        pos_pos2cls = {}
        pos_pos2depth = {}
        for direction in ["左边","左下方","左上方","正前方","正下方","正上方","右边","右下方","右上方"]:
            pos_pos2cls[direction] = {}
            pos_pos2depth[direction] = {}

        for i, num in enumerate(num_all):
            signal = 0
            if self.mode == 1:
                if num > 0:
                    if self.object_list[i] in ["person", "cup", "bowl", "banana", "apple", "laptop", "cell phone"]:
                        liangci = "个" 
                        signal  = 1
                    elif self.object_list[i] in ["diningtable", "chair"]:
                        liangci = "张"
                        signal  = 1
                    elif self.object_list[i] in ["book"]:
                        liangci = "本"
                        signal  = 1
                    elif self.object_list[i] in ["toothbrush", "spoon"]:
                        liangci = "支"
                        signal  = 1

            if self.mode == 2:
                if num > 0:
                    if self.object_list[i] in ["person"]:
                        liangci = "个"
                        signal = 1
                    elif self.object_list[i] in ["bicycle", "car", "motorbike", "bus", "truck"]:
                        liangci = "辆"
                        signal = 1

            if self.mode == 4:
                if num > 0:
                    if self.object_list[i] in ["person"]:
                        liangci = "个"
                        signal = 1
                    elif self.object_list[i] in ["diningtable", "chair"]:
                        liangci = "张"
                        signal = 1
                    elif self.object_list[i] in ["bicycle", "car", "motorbike", "bus", "truck"]:
                        liangci = "辆"
                        signal = 1    

            if self.mode == 7:
                if num > 0:
                    if self.object_list[i] in ["chair"]:
                        liangci = "张"
                        signal = 1
                        if num == 1:
                            adjust_flag = True

            if self.mode == 8:
                if num > 0:
                    if self.object_list[i] in ["cup"]:
                        liangci = "个" 
                        signal = 1
                        if num == 1:
                            adjust_flag = True

            if signal == 1:
                pos_outs = Counter(pos_cls2pos[i])
                pos_keys = pos_outs.keys()
                for pos in pos_keys:
                    if pos_outs[pos] >= 1:
                        pos_pos2cls[pos][self.object_dic[self.object_list[i]]] = str(pos_outs[pos]) + liangci
                        pos_pos2depth[pos][self.object_dic[self.object_list[i]]] = "距离为"
                        
                        for index, direction in enumerate(pos_cls2pos[i]):
                            if direction == pos:
                                if self.object_list[i] in ["chair"]:
                                    if self.mode == 7:
                                        if pos_cls2dep[i][index] < 0.8:
                                            return "已经到达椅子附近。"
                                        else:
                                            pos_cls2dep[i][index] = (pos_cls2dep[i][index]**2 - 0.8)**(1/2)
                                        
                                pos_pos2depth[pos][self.object_dic[self.object_list[i]]] += "%.2f"%pos_cls2dep[i][index] + "米。"

        #-------------------------------#
        #   创建文字
        #-------------------------------#
        for pos in pos_pos2cls.keys():
            if len(pos_pos2cls[pos]) != 0:
                sentence = sentence + pos + "有" 
                for classe in pos_pos2cls[pos].keys():
                    sentence = sentence + pos_pos2cls[pos][classe] + classe + "，"
                    if self.mode == 7 or self.mode == 8:
                        sentence = sentence + pos_pos2depth[pos][classe] + "。"
                    if adjust_flag:
                        if pos != "正前方":
                            sentence = sentence + "请往" + pos + "调整。"

        if sentence != "":
            sentence = "现在播报物体信息！识别到" + sentence
        print("目标信息如下：", self.object_class_all)
        return sentence

    def process_traffic(self):
        '''
        self.mode的值可选范围为：0,1,2,3,4,5,6,7,8
        0为无工作状态
        1为室内模式
        2为室外模式
        3为阅读模式
        4为障碍物识别模式
        5为交通灯识别模式
        6为交通标志牌识别模式
        7为椅子识别模式
        8为杯子识别模式
        '''
        sentence    = ""
        red_num     = 0
        green_num   = 0
        num_all     = np.zeros(len(self.traffic_list))

        #-------------------------------#
        #   判断每个类的数量
        #-------------------------------#
        for i, classe in enumerate(self.traffic_class_all):
            num_all[self.traffic_list.index(classe)] = num_all[self.traffic_list.index(classe)] + 1

        for i, num in enumerate(num_all):
            if self.mode == 2:
                if num > 0:
                    if self.traffic_list[i] in ["红灯"]:
                        red_num = num
                    elif self.traffic_list[i] in ["绿灯"]:
                        green_num = num
                    else:
                        sentence = sentence + self.traffic_list[i] + "标志。"

            if self.mode == 5:
                if num > 0:
                    if self.traffic_list[i] in ["红灯"]:
                        red_num = num
                    elif self.traffic_list[i] in ["绿灯"]:
                        green_num = num

            if self.mode == 6:
                if num > 0:
                    sentence = sentence + self.traffic_list[i] + "标志。"

        if red_num == 0 and green_num == 0 and sentence != "":
            sentence = sentence
        elif red_num >= 1 and green_num == 0:
            if sentence == "":
                sentence = str(int(red_num)) + "个红灯，注意安全！"
            else:
                sentence = str(int(red_num)) + "个红灯，注意安全！前方还有" + sentence
        elif red_num == 0 and green_num >= 1:
            if sentence == "":
                sentence = str(int(green_num)) + "个绿灯，通过马路注意安全！"
            else:
                sentence = str(int(green_num)) + "个绿灯，通过马路注意安全！前方还有" + sentence
        elif red_num >= 1 and green_num >= 1:
            if sentence == "":
                sentence = str(int(red_num)) + "个红灯和" + str(int(green_num)) + "个绿灯，无法正确判断是否可以前行，请向周围人群寻求帮助！"
            else:
                sentence = str(int(red_num)) + "个红灯和" + str(int(green_num)) + "个绿灯，无法正确判断是否可以前行，请向周围人群寻求帮助！前方还有" + sentence

        if sentence != "":
            sentence = "现在播报交通信息！识别到" + sentence
        print("交通信息如下：", self.traffic_class_all)
        return sentence

    def process_ocr(self):
        '''
        self.mode的值可选范围为：0,1,2,3,4,5,6,7,8
        0为无工作状态
        1为室内模式
        2为室外模式
        3为阅读模式
        4为障碍物识别模式
        5为交通灯识别模式
        6为交通标志牌识别模式
        7为椅子识别模式
        8为杯子识别模式
        '''
        sentence = ""
        #-------------------------------#
        #   判断是否存在书本
        #-------------------------------#
        num_all = np.zeros(len(self.object_list))
        #-------------------------------#
        #   创建全部的空列表
        #   判断每个类的数量以及位置
        #-------------------------------#
        pos_all = []
        for _ in num_all:
            pos_all.append([])

        for i, classe in enumerate(self.object_class_all):
            num_all[self.object_list.index(classe)] = num_all[self.object_list.index(classe)] + 1
            pos_all[self.object_list.index(classe)].append(self.computer_pos(i))

        for i, num in enumerate(num_all):
            signal = 0
            if num > 0:
                if self.object_list[i] in ["book"]:
                    sentence = sentence + str(int(num)) + "本" + self.object_dic[self.object_list[i]] + "。"
                    signal = 1

            if signal == 1:
                if "正前方" in pos_all[i]:
                    sentence = "一本书，在镜头正前方。"
                else:
                    for index in range(int(num)):
                        if num==1:
                            sentence = sentence + "在" + str(pos_all[i][index]) + "。"
                            sentence = sentence + "当前书本并非最佳识别位置，请把镜头往" + str(pos_all[i][index]) + "调整。"
                        else:
                            sentence = sentence + "一本在" + str(pos_all[i][index]) + "。"
                            sentence = sentence + "当前书本并非最佳识别位置，请注意调整到正前方。"

                sentence = "现在播报当页书本检测结果，识别到" + sentence

        if len(self.orc_save_sentence)>0:
            sentence = "已经完成本页文本识别，可以翻到下一页。" + sentence + "获得文字信息如下，" + self.orc_save_sentence
        else:
            sentence = "未识别到书本与文字，请注意调整书本的位置。"
        print("文字信息如下：", self.orc_save_sentence)
        return sentence

    def process_face(self):
        '''
        self.face_mode的值可选范围为：0,1,2,3
        0为无工作状态
        1为正常模式
        2为仅人脸识别
        3为表情识别
        '''
        sentence = ""
        unknow_num = 0
        for i, classe in enumerate(self.face_class_all):
            if self.face_mode == 1:
                name = classe.split(";")[0]
                if name == "未知":
                    unknow_num += 1
                    continue
                emotion = classe.split(";")[1]
                sentence = sentence + name + "在" + self.computer_face_pos(i) + "，表情为" + emotion + "。"

            if self.face_mode == 2:
                name = classe.split(";")[0]
                if name == "未知":
                    unknow_num += 1
                    continue
                sentence = sentence + name + "在" + self.computer_face_pos(i) + "。"

            if self.face_mode == 3:
                emotion = classe.split(";")[1]
                sentence = sentence + "一人在" + self.computer_face_pos(i) + "，表情为" + emotion + "。"

        if unknow_num > 0:
            sentence = sentence + str(unknow_num) + "张人脸未知。"

        # if len(self.face_class_all) >= 1:
        #     sentence = str(len(self.face_class_all)) + "张人脸。" + sentence

        if sentence != "":
            sentence = "现在播报人脸信息！识别到" + sentence

        print("人脸信息如下：", self.face_class_all)
        return sentence

    def process_pspnet(self):
        '''
        self.seg_mode的值可选范围为：0,1
        0为无工作状态
        1为开启斑马线识别
        '''
        sentence = ""
        for i, center in enumerate(self.pspnet_center):
            if self.seg_mode == 1:
                if center[0]==0 and center[1] == 0:
                    continue
                sentence = sentence + self.pspnet_class_all[i] + "在" + self.computer_pspnet_pos(i) + "。"

        if len(sentence) > 0:
            sentence = "现在播报道路信息！识别到" + sentence

        print("道路信息如下：", self.pspnet_center)
        return sentence

    def play_wav(self, dir, name, wake_signal=False):
        wf = wave.open(dir + "/" + name + ".wav", 'rb')
        chunk = 1024
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True)
        data = wf.readframes(chunk)
        if wake_signal:
            while len(data) > 0:
                stream.write(data)
                data = wf.readframes(chunk)
        else:
            while len(data) > 0:
                if self.wake_signal:
                    break
                stream.write(data)
                data = wf.readframes(chunk)
        stream.close()
        p.terminate()

    def mp3towav(self, dir, name):
        mp3 = AudioSegment.from_file(dir + "/" + name + ".mp3", format="mp3")
        # 修改对象参数
        wav = mp3.set_frame_rate(16000)
        wav = wav.set_channels(1)
        wav = wav.set_sample_width(2)
        # 导出wav文件
        wav.export(dir + "/" + name + ".wav", format='wav', )

    def record_voice(self, second):  # 记录声音
        rec = Recorder()
        # 记录五秒内的声音
        begin = time.time()
        print("Start recording")
        rec.start()
        time.sleep(second)
        print("Stop recording")
        rec.stop()
        # 保存
        fina = time.time()
        t = fina - begin
        print('录音时间为%ds' % t)
        rec.save("mp3andwav/order.wav")  # 存取录音

        # 判断录音的内容
        word_dict = client.asr(get_file_content("mp3andwav/order.wav"), "wav", 16000, {
            'dev_pid': 1537,
        })
        final_result = ""
        # 如果录音为空则无
        if word_dict["err_no"] != 0:
            return final_result

        word_list = word_dict["result"]
        # 打印录到的内容
        for i in word_list:
            final_result = final_result + i + " "
        print(final_result)
        return final_result

    def play_sentence(self, sentence, name="sentence", wake_signal=False):
        # ----------------------------#
        #   转为mp3
        # ----------------------------#
        result = client_for_speak.synthesis(sentence, 'zh', 2, {
            'vol': 5, 'per': 0, 'spd': self.speck_speed
        })
        if not isinstance(result, dict):
            with open('mp3andwav/' + name  +'.mp3', 'wb') as f:
                f.write(result)
        else:
            time.sleep(1)
            return
        # ----------------------------#
        #   mp3转wav
        # ----------------------------#
        self.mp3towav("mp3andwav", name)

        # ----------------------------#
        #   播放
        # ----------------------------#
        self.play_wav("mp3andwav", name, wake_signal = wake_signal)

    def wake_up(self):
        # 判断应该怎么搞
        self.wake_signal = True
        sentence = "请说！"
        try:
            t1 = threading.Thread(target=run_send_sentence, args=(sentence, self.urls['sentence_url']))
            t1.start()
        except:
            pass
        # ----------------------------#
        #   播放
        # ----------------------------#
        if os.path.exists("mp3andwav/IamHere.mp3"):
            self.play_wav("mp3andwav", "IamHere", wake_signal=True)
        else:
            self.play_sentence(sentence,"IamHere", wake_signal=True)

        final_result = self.record_voice(4)

        # 判断应该怎么搞
        sentence = "请说出正确的模式"
        try:
            # ----------------------------#
            #   转为mp3
            # ----------------------------#
            if final_result.find(u"录入") >= 0:
                self.mode = 0
                self.face_mode = 0
                self.seg_mode = 0
                # ----------------------------#
                #   播放
                # ----------------------------#
                if os.path.exists("mp3andwav/say_name.mp3"):
                    self.play_wav("mp3andwav", "say_name", wake_signal=True)
                else:
                    self.play_sentence("请说出人脸名称","say_name", wake_signal=True)

                final_result = self.record_voice(3)
                
                img_base64 = image_to_base64(self.image)
                dataPModel = {"img_base64": img_base64}
                dataPModel = {"name": final_result}

                result = json_send(dataPModel,self.face_record_url)['result']

                bounding_boxes, class_all = result["face_locations"], result["class_all"]

                if len(bounding_boxes)!=1:
                    sentence = "存在多个人脸或者不存在人脸"
                else:
                    sentence = "已录入"

            if final_result.find(u"关闭所有") >= 0:
                self.mode = 0
                self.face_mode = 0
                self.seg_mode = 0
                sentence = "关闭所有功能"
                print("关闭所有功能")

            if final_result.find(u"关闭室内导航") >= 0:
                self.navigation_mode = 0
                self.navigation_place = 0
                img_base64 = image_to_base64(self.image)
                dep_img_base64 = image_to_base64_dep(self.depth*10)

                dataPModel = {"img_base64": img_base64}
                dataPModel["dep_img_base64"] = dep_img_base64
                dataPModel["navigation_mode"] = self.navigation_mode
                dataPModel["navigation_place"] = self.navigation_place
                sentence = "关闭室内导航"
                print("关闭室内导航")

            if final_result.find(u"关闭室内模式") >= 0:
                self.mode = 0
                sentence = "关闭室内模式"
                print("关闭室内模式")

            if final_result.find(u"关闭室外") >= 0:
                self.mode = 0
                sentence = "关闭室外模式"
                print("关闭室外模式")

            if final_result.find(u"关闭文字") >= 0:
                self.mode = 0
                sentence = "关闭文字识别"
                print("关闭文字识别")

            if final_result.find(u"关闭人脸") >= 0:
                self.face_mode = 0
                sentence = "关闭人脸识别"
                print("关闭人脸识别")

            if final_result.find(u"关闭道路") >= 0:
                self.seg_mode = 0
                sentence = "关闭道路识别"
                print("关闭道路识别")

            if final_result.find(u"开启室内导航") >= 0 or final_result.find(u"打开室内导航") >= 0 or final_result.find(u"开始室内导航") >= 0:
                self.navigation_place = 0
                # ----------------------------#
                #   播放
                # ----------------------------#
                if os.path.exists("mp3andwav/say_place_name.mp3"):
                    self.play_wav("mp3andwav", "say_place_name", wake_signal=True)
                else:
                    self.play_sentence("请说出地点名称","say_place_name", wake_signal=True)

                try:
                    sentence = "请说出地点名称"
                    t1 = threading.Thread(target=run_send_sentence, args=(sentence, self.urls['sentence_url']))
                    t1.start()
                except:
                    pass
                final_result = self.record_voice(3)
                
                if final_result.find(u"会议室") >= 0:
                    self.navigation_place = 1
                    place = "会议室"

                if self.navigation_place == 0:
                    self.navigation_mode = 0
                    sentence = "请说出正确的目标点"
                else:
                    self.navigation_mode = 1
                    sentence = "已经打开室内导航，导航到" + place

                print("打开室内导航")

            if final_result.find(u"开启室内模式") >= 0 or final_result.find(u"打开室内模式") >= 0 or final_result.find(u"开始室内模式") >= 0:
                self.mode = 1
                sentence = "打开室内模式"
                print("打开室内模式")

            if final_result.find(u"开启室外") >= 0 or final_result.find(u"打开室外") >= 0 or final_result.find(u"开始室外") >= 0:
                self.mode = 2
                sentence = "打开室外模式"
                print("打开室外模式")

            if final_result.find(u"开启文字") >= 0 or final_result.find(u"打开文字") >= 0 or final_result.find(u"开始文字") >= 0:
                self.mode = 3
                self.face_mode = 0
                self.seg_mode = 0
                sentence = "打开文字识别"
                print("打开文字识别")

            if final_result.find(u"开启人脸") >= 0 or final_result.find(u"打开人脸") >= 0 or final_result.find(u"开始人脸") >= 0:
                self.face_mode = 2
                sentence = "打开人脸识别"
                print("打开人脸识别")

            if final_result.find(u"开启道路识别") >= 0 or final_result.find(u"打开道路识别") >= 0 or final_result.find(u"开始道路识别") >= 0:
                self.seg_mode = 1
                sentence = "打开道路识别"
                print("打开道路识别")

            if final_result.find(u"寻找椅子") >= 0:
                self.mode = 7
                sentence = "开始寻找椅子"
                print("开始寻找椅子")

            if final_result.find(u"寻找杯子") >= 0:
                self.mode = 8
                sentence = "开始寻找杯子"
                print("开始寻找杯子")
        except:
            pass
        
        # ----------------------------#
        #   播放
        # ----------------------------#
        try:
            t1 = threading.Thread(target=run_send_sentence, args=(sentence, self.urls['sentence_url']))
            t1.start()
        except:
            pass

        if os.path.exists("mp3andwav/" + sentence + ".mp3"):
            self.play_wav("mp3andwav", sentence, wake_signal=True)
        else:
            self.play_sentence(sentence, sentence, wake_signal=True)
        self.recognize_init()
        self.wake_signal = False

    def play_sound(self):
        sentence = ""
        if self.wake_signal:
            return
        if self.mode == 0 and self.face_mode == 0 and self.seg_mode == 0 and self.navigation_mode == 0:
            return
            
        if self.mode == 2 or self.mode == 5 or self.mode == 6:
            sentence += self.process_traffic()

        if self.mode == 1 or self.mode == 2 or self.mode == 4 or self.mode==7 or self.mode==8:
            sentence += self.process_object()

        if self.face_mode == 1 or self.face_mode == 2 or self.face_mode == 3:
            sentence += self.process_face()

        if self.mode == 3:
            sentence += self.process_ocr()

        if self.seg_mode == 1:
            sentence += self.process_pspnet()

        if self.navigation_end:
            self.navigation_mode = 0
            self.navigation_place = 0
            self.navigation_end = 0
            sentence += "导航结束！"
            
        if self.navigation_mode == 1:
            sentence += self.navigation_sentence

        if sentence != "":
            self.speak_flag = True
        else:
            sentence = "未检测到内容"

        if self.speak_flag:
            try:
                t1 = threading.Thread(target=run_send_sentence, args=(sentence, self.urls['sentence_url']))
                t1.start()
            except:
                pass

            if len(sentence)>12:
                self.play_sentence(sentence, "sentence")
            else:
                if os.path.exists("mp3andwav/" + sentence + ".mp3"):
                    self.play_wav("mp3andwav", sentence)
                else:
                    self.play_sentence(sentence, sentence)
            # self.play_sentence(sentence, "recognize")
        
        if sentence == "未检测到内容":
            self.speak_flag = False


    def save_images(self, image, start_Time):
        '''
        self.mode的值可选范围为：0,1,2,3,4,5,6,7
        0为无工作状态
        1为室内模式
        2为室外模式
        3为阅读模式
        4为障碍物识别模式
        5为交通灯识别模式
        6为交通标志牌识别模式
        7为盲道和斑马线识别
        self.face_mode的值可选范围为：0,1,2,3
        0为无工作状态
        1为正常模式
        2为仅人脸识别
        3为表情识别
        '''
        path = "./Saves/photo/" + start_Time + "/"
        file = open("./Saves/txt/" + start_Time + ".txt", "a")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(np.uint8(image))

        nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        classes = []
        bounding_boxes = []
        if self.mode == 1:
            for i,j in zip(self.object_class_all,self.object_bounding_boxes):
                if i in mode_classes[self.mode]:
                    classes.append(i)
                    bounding_boxes.append(j)

        elif self.mode == 2:
            for i,j in zip(self.object_class_all,self.object_bounding_boxes):
                if i in mode_classes[self.mode]:
                    classes.append(i)
                    bounding_boxes.append(j)
                    
            for i,j in zip(self.traffic_class_all,self.traffic_bounding_boxes):
                classes.append(i)
                bounding_boxes.append(j)

        elif self.mode == 3:
            if (self.mode != 0 or self.face_mode != 0) and self.wake_signal == False:
                image.save(path + nowTime + ".jpg")
                print(self.orc_save_sentence)
                file.write(nowTime + " : " + self.orc_save_sentence + "\n")
                return
                
        elif self.mode == 4 or self.mode==7 or self.mode==8:
            for i,j in zip(self.object_class_all,self.object_bounding_boxes):
                if i in mode_classes[self.mode]:
                    classes.append(i)
                    bounding_boxes.append(j)

        elif self.mode == 5:
            for i,j in zip(self.traffic_class_all,self.traffic_bounding_boxes):
                classes.append(i)
                bounding_boxes.append(j)

        elif self.mode == 6:
            for i,j in zip(self.traffic_class_all,self.traffic_bounding_boxes):
                classes.append(i)
                bounding_boxes.append(j)
                
        if self.face_mode == 1:
            try:
                for i,j in zip(self.face_class_all,self.face_bounding_boxes[:, :4]):
                    classes.append(i)
                    bounding_boxes.append(j)
            except:
                pass
        elif self.face_mode == 2:
            try:
                for i,j in zip(self.face_class_all,self.face_bounding_boxes[:, :4]):
                    i = i.split(";")[0]
                    classes.append(i)
                    bounding_boxes.append(j)
            except:
                pass
        elif self.face_mode == 3:
            try:
                for i,j in zip(self.face_class_all,self.face_bounding_boxes[:, :4]):
                    i = i.split(";")[-1]
                    classes.append(i)
                    bounding_boxes.append(j)
            except:
                pass
        
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = 5

        if (self.mode != 0 or self.face_mode != 0) and self.wake_signal == False:
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
