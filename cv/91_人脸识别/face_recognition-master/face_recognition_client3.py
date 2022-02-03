import json
from PIL import Image, ImageDraw
import numpy as np
import requests
import base64

def image_bytes(file):  #--same with read_gfile
    with open(file, "rb") as imageFile:
        ib = base64.b64encode(imageFile.read()).decode()
        return ib

b64 = image_bytes("D:\\1.jpg")

data = json.dumps({'instances': [{"b64": b64}]})
# print(data)

headers = {"content-type": "application/json"}
json_response = requests.post('http://172.27.152.172:5001/v1/models/fr_face_encodings/versions/1/',
        data=data, headers=headers)

predictions = json.loads(json_response.text)['predictions']
print(predictions)
