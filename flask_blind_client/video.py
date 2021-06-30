#-------------------------------------#
#       调用摄像头检测
#-------------------------------------#
#realsense摄像头的操作，能生成两种图片：彩色图和深度图
import time

import cv2
import numpy as np

import pyrealsense2 as rs
from utils.utils import image_to_base64, image_to_base64_dep

print("正在初始化摄像头")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)#bgr格式
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)#深度图
profile = pipeline.start(config)

align = rs.align(rs.stream.color)#深度图对齐到RGB，这里可以弄懂一下？

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()#获取深度传感器深度位数

frames = pipeline.wait_for_frames()#
aligned_frames = align.process(frames)# 对齐后帧处理
aligned_depth_frame = aligned_frames.get_depth_frame()
aligned_color_frame = aligned_frames.get_color_frame()

assert aligned_depth_frame and aligned_color_frame#断言什么pip
depth_image = np.asanyarray(aligned_depth_frame.get_data(),dtype=np.float32)
image = np.asanyarray(aligned_color_frame.get_data())
depth_map = depth_image * depth_scale

# 调用摄像头
index = 0

while(True):
    a = time.time()
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    aligned_color_frame = aligned_frames.get_color_frame()

    if not aligned_depth_frame or not aligned_color_frame:
        continue

    depth_image = np.asanyarray(aligned_depth_frame.get_data(),dtype=np.uint16)
    color_image = np.asanyarray(aligned_color_frame.get_data())
    depth_map = depth_image * depth_scale
    
    # 通过cv2.cvtColor把图像从BGR转换到HSV
    img_hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 0] = (np.maximum(img_hsv[:, :, 0]+10,0)) % 180#这里是为了啥
    color_image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    # print(np.max(depth_map))
    img_base64 = image_to_base64(color_image)
    dep_img_base64 = image_to_base64_dep(depth_map*10)
    
    cv2.imshow("video",color_image)
    c = cv2.waitKey(30) & 0xff 
    print(time.time()-a)
    if c==32:#空格键
        cv2.imwrite("./photo_get/"+str(index)+"RGB.jpg",color_image,)#图像处理这块知识要补习一下
        cv2.imwrite("./photo_get/"+str(index)+"Depth.jpg",np.uint8(depth_map*10),)
        index += 1
