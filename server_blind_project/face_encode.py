import os

import cv2
import face_recognition
import numpy as np
from PIL import Image

from face_work.facenet_nets.arcface import arcface
from face_work.retinaface import Retinaface
from face_work.utils.utils import Alignment_1

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def face_encoder():
    # 读取数据库
    names = os.listdir("face_dataset")
    known_face_encodings = []
    known_face_names = []
    for name in names:
        jpgs = os.listdir("face_dataset/"+name)
        for jpg in jpgs:
            # try:
                image = np.array(Image.open("./face_dataset/" +name + "/" + jpg))
                height, width, _ = np.shape(image)

                # 彩色图像
                rgb_small_frame = image
                # 根据上述参数进行人脸检测
                face_locations = retinaface.detect_image(rgb_small_frame)

                if len(face_locations) == 0:
                    print(jpg)
                    continue

                face_locations = np.array(face_locations, dtype=np.int32)
                face_locations[:, [0,2]] = np.clip(face_locations[:, [0,2]], 0, width)
                face_locations[:, [1,3]] = np.clip(face_locations[:, [1,3]], 0, height)

                results = np.array(face_locations)

                biggest_area = 0
                best_face_location = None
                for result in results:
                    left, top, right, bottom = result[0:4]
                    w = right - left
                    h = bottom - top
                    if w*h > biggest_area:
                        biggest_area = w*h
                        best_face_location = result
                # 截取图像
                crop_img    = rgb_small_frame[int(best_face_location[1]):int(best_face_location[3]), int(best_face_location[0]):int(best_face_location[2])]
                # if name=="李国欣":
                #     cv2.imshow("123",crop_img)
                #     cv2.waitKey(0)
                landmark    = np.reshape(best_face_location[5:],(5,2)) - np.array([int(best_face_location[0]),int(best_face_location[1])])
                crop_img,_  = Alignment_1(crop_img,landmark)                
                crop_img    = np.array(letterbox_image(Image.fromarray(np.uint8(crop_img)),(input_shape[0],input_shape[1])))/255
                crop_img    = np.expand_dims(crop_img,0)
                
                # 利用facenet_model计算128维特征向量
                face_encoding = facenet.predict(crop_img)
                known_face_encodings.append(face_encoding[0])
                known_face_names.append(name)
                print("./face_dataset/" +name + "/" + jpg)
            # except:
            #     print(jpg)
            #     pass

    np.save("model_data/known_face_encodings.npy",known_face_encodings)
    np.save("model_data/known_face_names.npy",known_face_names)

if __name__ == "__main__":
    retinaface = Retinaface()
    input_shape = [112,112,3]
    facenet = arcface(input_shape)
    facenet.load_weights("./model_data/arcface_weights.h5")
    face_encoder()
