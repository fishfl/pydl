import json
import numpy as np
import requests
import base64
import matplotlib.pyplot as plt


def image_bytes(file):  # --same with read_gfile
    with open(file, "rb") as imageFile:
        ib = base64.b64encode(imageFile.read()).decode()
        return ib


b64 = image_bytes("D:\\2.jpg")
data = json.dumps({'instances': [{"b64": b64}]})
# print(data)

headers = {"content-type": "application/json"}
json_response = requests.post('http://172.27.148.34:5001/v1/models/pe_holistic_keypoints/versions/1/',
                              data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']
print(predictions)

json_response = requests.post('http://172.27.148.34:5001/v1/models/pe_holistic_image/versions/1/',
                              data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']
img = np.array(predictions)
plt.imshow(img)
plt.show()
