import sys
sys.path.append("/home/nvidia/XJQ_WORK/Flask-for-blind-module/")#把路径加入环境，可以直接import路径下的文件
sys.path.append("/home/nvidia/XJQ_WORK/Flask-for-blind-module/Thirdparty/snowboy/examples/Python3/")
from detect import Detect

import os
import cv2
import time
import signal
import threading
import numpy as np
import snowboydecoder
import datetime
import pyrealsense2 as rs

class Integrate_Model():
    def __init__(self, image, depth_map, detect, pipeline, align, depth_scale):
        self.colorimage             = image#彩色图
        self.image_for_recognize    = image#彩色图做目标识别
        self.depth_map              = depth_map#深度图
        self.detect                 = detect
        self.pipeline               = pipeline# 图片数据传送管道
        self.align                  = align#
        self.depth_scale            = depth_scale#

    def get_image(self):
        try:#异常处理模块
            frames                  = self.pipeline.wait_for_frames()#new frame
            aligned_frames          = self.align.process(frames)

            aligned_depth_frame     = aligned_frames.get_depth_frame()
            aligned_color_frame     = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not aligned_color_frame:
                return

            depth_image             = np.asanyarray(aligned_depth_frame.get_data(),dtype=np.float32)
            color_image             = np.asanyarray(aligned_color_frame.get_data())
            depth_map               = depth_image * self.depth_scale

            img_hsv                 = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            img_hsv[:, :, 0]        = (np.maximum(img_hsv[:, :, 0]+10,0)) % 180
            color_image             = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
            
            self.colorimage         = color_image
            self.depth_map          = depth_map

            video.write(self.colorimage)
            cv2.imshow("image", self.colorimage)
            cv2.waitKey(1)

        except Exception as ex:
            print("出现如下异常%s"%ex)
            time.sleep(1)

    def rec_and_play(self):
        try:
            temp_image  = self.colorimage
            temp_dep    = self.depth_map

            detect.recognize_all(temp_image, temp_dep)
            detect.play_sound()
            detect.save_images(temp_image, nowTime)
            time.sleep(1)

        except Exception as ex:
            print("出现如下异常%s"%ex)
            time.sleep(1)#sleep啥意义？

    def send_images(self):
        try:
            temp_image  = self.colorimage
            temp_dep    = self.depth_map
            detect.send_images(temp_image, temp_dep)#detect

        except Exception as ex:
            print("出现如下异常%s"%ex)
            time.sleep(1)

# ----------------------------#
#   识别
# ----------------------------#
def rec_image(integrate_model):#录制和播放
    while 1:
        integrate_model.rec_and_play()
        
# ----------------------------#
#   用于时刻获取图像
# ----------------------------#
def get_image(integrate_model):
    while 1:
        integrate_model.get_image()


def send_image(integrate_model):
    while 1:
        integrate_model.send_images()

# ----------------------------#
#   听到小白后的唤醒内容
# ----------------------------#
def signal_handler():
    detect.wake_up()

def interrupt_callback():
    global interrupted#全局变量
    return interrupted

if __name__ == "__main__":
    interrupted = False
    #--------------------------------------------------------#
    #   建立文件夹
    #--------------------------------------------------------#
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    os.makedirs("./Saves/photo/" + nowTime)
    
    #--------------------------------------------------------#
    #   初始化摄像头部分
    #--------------------------------------------------------#
    print("正在初始化摄像头")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    aligned_color_frame = aligned_frames.get_color_frame()
    assert aligned_depth_frame and aligned_color_frame

    depth_image = np.asanyarray(aligned_depth_frame.get_data(),dtype=np.float32)
    image = np.asanyarray(aligned_color_frame.get_data())
    depth_map = depth_image * depth_scale

    height, width, _ = np.shape(image)#图片的高和宽

    #take frames into videos
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter("Saves/video/" + nowTime + '.avi', fourcc, 6.0, (width, height))#将帧做成视频

    #--------------------------------------------------------#
    #   url路径
    #--------------------------------------------------------#
    # urls = {
    #     'object_url'      : 'http://justonetest.qicp.vip:80/object_detection',
    #     'traffic_url'     : 'http://justonetest.qicp.vip:80/traffic_detection',
    #     'face_url'        : 'http://justonetest.qicp.vip:80/face_recognize',
    #     'face_record_url' : 'http://justonetest.qicp.vip:80/face_record',
    #     'pspnet_url'      : 'http://justonetest.qicp.vip:80/segmentation',
    #     'sentence_url'    : 'http://justonetest.qicp.vip:80/play_voice',
    #     'send_image_url'  : 'http://justonetest.qicp.vip:80/get_image',
    # }
    #urls算是网络里面的路径，我访问到这里
    urls = {
        'object_url'      : 'http://192.168.1.110:8502/object_detection',
        'traffic_url'     : 'http://192.168.1.110:8502/traffic_detection',
        'face_url'        : 'http://192.168.1.110:8502/face_recognize',
        'face_record_url' : 'http://192.168.1.110:8502/face_record',
        'pspnet_url'      : 'http://192.168.1.110:8502/segmentation',
        'send_image_url'  : 'http://192.168.1.110:8502/get_image2',
        'sentence_url'    : 'http://192.168.1.115:8502/play_voice',
    }
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
    mode = 0
    seg_mode = 0
    face_mode = 0
    navigation_mode = 0
    navigation_place = 1 
    #---------------------------#
    #   初始化全局detect
    #---------------------------#
    print("正在初始化全局Detect")
    detect = Detect(height, width, mode=mode, face_mode=face_mode, 
        seg_mode=seg_mode, navigation_mode=navigation_mode, navigation_place=navigation_place, urls=urls)

    #---------------------------#
    #   将图片读取和播放模块结合起来
    #---------------------------#
    print("正在初始化结合播放模块")
    integrate_model = Integrate_Model(image, depth_map, detect, pipeline, align, depth_scale)

    #-----------------------------------#
    #   这两个线程用来控制播放模块与初始化模块
    #-----------------------------------#

    #多线程的一个重要意义在于防止阻塞，一个线程阻塞不影响另外的线程
    print("开启识别线程")
    rec_image_thread = threading.Thread(target=rec_image,args=(integrate_model,))
    rec_image_thread.start()

    print("开启图像获取线程")
    get_image_thread = threading.Thread(target=get_image,args=(integrate_model,))
    get_image_thread.start()

    print("开启图像传输线程")
    send_image_thread = threading.Thread(target=send_image,args=(integrate_model,))
    send_image_thread.start()

    #-----------------------------------#
    #   语音提醒已经开启所有线程
    #-----------------------------------#
    detect.wake_signal = True
    print("线程已经全部开启")
    if os.path.exists("mp3andwav/init.mp3"):
        detect.play_wav("mp3andwav", "init", wake_signal=True)
    else:
        detect.play_sentence("初始化完成，开始接受指令。", "init", wake_signal=True)
    detect.wake_signal = False

    print("已经开启语音唤醒功能")
    model = "model_data/xiaobai.pmdl"
    signal.signal(signal.SIGINT, signal_handler)
    #snowboydecoder在哪
    detector = snowboydecoder.HotwordDetector(model, sensitivity=0.5)#这个函数不在程序里面
    print('正在聆听... Press Ctrl+C to exit')
    li = detector.start(detected_callback=signal_handler,
                        interrupt_check=interrupt_callback,
                        sleep_time=0.03)

    detector.terminate()
