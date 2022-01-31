import json
import os
from PIL import Image
import numpy as np
import requests
import io
import base64
import tensorflow as tf

def image_bytes(file):  #--same with read_gfile
    with open(file, "rb") as imageFile:
        ib = base64.b64encode(imageFile.read()).decode()
        return ib

def read_gfile(file):
    image_data = tf.compat.v1.gfile.FastGFile(file, 'rb').read()
    ib = base64.b64encode(image_data).decode()
    return ib

def read_io(file):   # --this work too
    i = Image.open(file)
    imgByteArr = io.BytesIO()
    i.save(imgByteArr, format='JPEG')
    imgByteArr = imgByteArr.getvalue()
    ib = base64.b64encode(imgByteArr).decode()
    return ib

def read_byte(file):
    i = Image.open(file)
    img = np.array(i)
    return img

i = image_bytes("D:\\test\\4.jpg")

# data = '{"signature_name": "serving_default", "instances": [{"b64": "%s"}]}' % i   no work
data = '{"signature_name": "serving_default", "inputs": {"in" : {"b64": "%s"}}}' % i
# print(data)

headers = {"content-type": "application/json"}
json_response = requests.post('http://172.27.148.129:8501/v1/models/porn_verify:predict',
        data=data, headers=headers)

predictions = json.loads(json_response.text)
print(predictions)
