
from io import BytesIO
from PIL import Image
from time import time
import requests
import datetime
import base64
import json
import re

def image_to_base64(img):
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str.decode()

def base64_to_image(base64_str):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    return img
    
def base64_to_dep_image(base64_str):
    # 从base64编码恢复numpy数组
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    str_decode = base64.b64decode(base64_data)
    img = np.fromstring(str_decode, np.float32)
    return img