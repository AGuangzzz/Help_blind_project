import datetime
import json
import sys

from keras.models import Model

import rospy
from std_msgs.msg import String

sys.path.insert(1, '/home/robot/catkin_workspace/install/lib/python3/dist-packages')
import threading
import time
from socket import *


import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from flask import Flask, request
from flask_cors import *
from sensor_msgs.msg import Image

from addition_work.pspnet import Pspnet        #pspnet
from detect import Detect
from face_work.face_rec import face_rec
from object_work.yolo import YOLO as object_YOLO
from ocr_work.ocr import Ocr_module
from traffic_work.yolo import YOLO as traffic_YOLO
from utils import base64_to_dep_image, base64_to_image, image_to_base64

app = Flask(__name__)
CORS(app)

@app.route('/object_detection',methods=['GET','POST'])
def object_detection():     #目标检测模块

    #得到客户端传输的字典数据：  客户端传送过来的json数据，request的请求和响应
    img_base64 = request.json['img_base64']
    mode        = request.json['mode']
    face_mode   = request.json['face_mode']      #获取face_mode

    #转换成array格式
    img = np.array(base64_to_image(img_base64))
    print("Get_images. Start Object_detection")
    bounding_boxes, class_all, traffic_boxes, traffic_class, ids = object_yolo.detect_image(img)  #object_yolo.detect_image获取五个参数
    print("Detect_image done")

    if mode == 2:
        _, _, traffic_boxes, traffic_class, _ = traffic_yolo.detect_image(img)

    # 返回内容，返回json格式
    jsonresult = {}
    #转换成列表格式
    jsonresult['bounding_boxes'] = bounding_boxes.tolist()
    jsonresult['class_all']      = class_all.tolist()
    jsonresult['traffic_boxes']  = traffic_boxes.tolist()
    jsonresult['traffic_class']  = traffic_class.tolist()
    if mode == 3:
        book_num = 0
        for small_cls in list(class_all):
            if small_cls == "book":
                book_num += 1

        if book_num == 1:
            if "book" in list(class_all):
                left, top, right, bottom    = bounding_boxes[list(class_all).index("book")]
                img                         = img[top:bottom,left:right]
            
        orc_save_sentence                   = ocr_module.get_ocr(img)
        jsonresult['orc_save_sentence']     = orc_save_sentence
        
        txt_file                            = open("./results/txt/" + nowTime + ".txt", "a")
        txt_file.write(orc_save_sentence + "\n")
        txt_file.close()

    txt_file    = open("./results/txt/" + nowTime + ".txt", "a")
    detect.save_images(img,"./results/photo/",txt_file, bounding_boxes, class_all, traffic_boxes, traffic_class, None, None, mode=mode,face_mode=face_mode)
    txt_file.close()

    #返回格式，result
    dataPModel = {"msg": "object_detection success", "code": 200, "result": jsonresult}
    
    return json.dumps(dataPModel)

@app.route('/traffic_detection',methods=['GET','POST'])
def traffic_detection():
    img_base64 = request.json['img_base64']
    mode        = request.json['mode']
    face_mode   = request.json['face_mode']

    img = np.array(base64_to_image(img_base64))
    print("Get_images. Start Traffic_detection")
    bounding_boxes, class_all, traffic_boxes, traffic_class, ids = traffic_yolo.detect_image(img)
    print("Detect_image done")

    # 返回内容，返回json格式
    jsonresult = {}
    jsonresult['bounding_boxes'] = bounding_boxes.tolist()
    jsonresult['class_all']      = class_all.tolist()
    jsonresult['traffic_boxes']  = traffic_boxes.tolist()
    jsonresult['traffic_class']  = traffic_class.tolist()

    txt_file    = open("./results/txt/" + nowTime + ".txt", "a")
    detect.save_images(img,"./results/photo/",txt_file, bounding_boxes, class_all, traffic_boxes, traffic_class, None, None, mode=mode,face_mode=face_mode)
    txt_file.close()

    dataPModel = {"msg": "traffic_detection success", "code": 200, "result": jsonresult}
    
    return json.dumps(dataPModel)

@app.route('/face_recognize',methods=['GET','POST'])
def face_recognize():
    img_base64  = request.json['img_base64']
    mode        = request.json['mode']
    face_mode   = request.json['face_mode']

    img = np.array(base64_to_image(img_base64))
    print("Get_images. Start Face_recognize")
    face_locations, class_all, ids = face.recognize(img)
    print("Detect_image done")

    # 返回内容，返回json格式
    jsonresult = {}
    jsonresult['class_all']      = class_all.tolist()
    jsonresult['face_locations'] = face_locations.tolist()

    txt_file    = open("./results/txt/" + nowTime + ".txt", "a")
    detect.save_images(img,"./results/photo/",txt_file, None, None, None, None, face_locations, class_all, mode=mode,face_mode=face_mode)
    txt_file.close()

    dataPModel = {"msg": "face_recognize success", "code": 200, "result": jsonresult}
    return json.dumps(dataPModel)

@app.route('/face_record',methods=['GET','POST'])
def face_record():
    img_base64 = request.json['img_base64']
    name = request.json['name']

    img = np.array(base64_to_image(img_base64))
    print("Get_images. Start Facerecording")
    face_locations, class_all, ids = face.add_face(img,name)
    print("Detect_image done")

    # 返回内容，返回json格式
    jsonresult = {}
    jsonresult['class_all']      = class_all.tolist()
    jsonresult['face_locations'] = face_locations.tolist()
    dataPModel = {"msg": "face_record success", "code": 200, "result": jsonresult}
    return json.dumps(dataPModel)

@app.route('/segmentation',methods=['GET','POST'])
def segmentation():
    img_base64 = request.json['img_base64']

    img = np.array(base64_to_image(img_base64))
    print("Get_images. Start Segmentation")
    bounding_boxes, centers, class_all = pspnet.detect_image(img)       #detect_image 来自PSPnet.py,返回np.array(bboxes), np.array(centers), np.array(["盲道", "斑马线"])
    print("Detect_image done")

    # 返回内容，返回json格式
    jsonresult = {}
    jsonresult['bounding_boxes'] = bounding_boxes.tolist()
    jsonresult['centers']        = centers.tolist()
    jsonresult['class_all']      = class_all.tolist()  
    dataPModel = {"msg": "segmentation success", "code": 200, "result": jsonresult}
    return json.dumps(dataPModel)

@app.route('/play_voice',methods=['GET','POST'])
def play_voice():
    sentence = request.json['sentence']
    
    jsonresult = {}
    jsonresult['reply'] = "OK"
    dataPModel = {"msg": "play_voice success", "code": 200, "result": jsonresult}
    return json.dumps(dataPModel)


@app.route('/get_image',methods=['GET','POST'])
def get_image():
    print("connect_get")
    global a
    b = time.time()
    img_base64 = request.json['img_base64']
    dep_img_base64  = request.json['dep_img_base64']
    navigation_mode  = request.json['navigation_mode']
    navigation_place  = str(request.json['navigation_place'])

    image = np.array(base64_to_image(img_base64),np.uint8)
    dep_image = np.array(np.array(base64_to_image(dep_img_base64))/10*1000, np.uint16)

    cv2.imwrite("Now_U_See_me.jpg",image)
    cv2.imwrite("Now_U_See_me_dep.jpg",dep_image)
    print("get_image")
    image_pubulish.publish(bridge.cv2_to_imgmsg(image, "bgr8"))
    print("get_dep_image")
    dep_image_pubulish.publish(bridge.cv2_to_imgmsg(dep_image, "16UC1"))
    
    tcpCliSock.send(navigation_place.encode())
    words = tcpCliSock.recv(BUFSIZE).decode()

    # 返回内容，返回json格式S
    jsonresult = {}
    jsonresult['reply'] = "OK"
    jsonresult['navigation_end'] = int(words.split(";")[1])
    jsonresult['navigation_sentence'] = words.split(";")[0]
    print(words)
    print(time.time()-a)
    dataPModel = {"msg": "get_image success", "code": 200, "result": jsonresult}
    a = time.time()
    return json.dumps(dataPModel)

def talker():
    pub = rospy.Publisher('chatter', String, queue_size=1)

    rate = rospy.Rate(30)
    
    while not rospy.is_shutdown():
        pub.publish("hello_str")
        rate.sleep()

if __name__ == '__main__':
    detect          = Detect()

    object_yolo     = object_YOLO()
    print("已载入目标检测算法")
    traffic_yolo    = traffic_YOLO()
    print("已载入交通信息检测算法")
    face            = face_rec()
    print("已载入人脸识别算法")
    ocr_module      = Ocr_module()
    print("已载入文字识别算法")
    pspnet          = Pspnet()
    print("已载入语义分割算法")

    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')

    #ros这一块的内容还需要学习一下。
    rospy.init_node("Publisher", anonymous=True)
    print("ROS节点初始化")

    bridge = CvBridge()
    print("载入ROS的CV桥")
    image_pubulish=rospy.Publisher('/D435i/color/image_raw', Image, queue_size=1)
    dep_image_pubulish=rospy.Publisher('/D435i/depth/image_rect_raw', Image, queue_size=1)
    print("载入ROS的分发主题")
    a = time.time()

    thread_talk = threading.Thread(target=talker)
    thread_talk.start()
    
    BUFSIZE = 1024
    ADDR = ('127.0.0.1', 6682)
    tcpCliSock = socket(AF_INET,SOCK_STREAM)
    tcpCliSock.connect(ADDR)
    print("SOCKET连接")

    # app.run(host="192.168.8.112", port=8502,use_reloader=False)
    # app.run(host="127.0.0.1", port=8502, use_reloader=False)
    app.run(host="192.168.124.65", port=8502, use_reloader=False)   #进行服务器连接

