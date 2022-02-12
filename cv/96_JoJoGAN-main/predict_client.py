import json
import numpy as np
import requests
import base64
import matplotlib.pyplot as plt


def image_bytes(file):  # --same with read_gfile
    with open(file, "rb") as imageFile:
        ib = base64.b64encode(imageFile.read()).decode()
        return ib


b64 = image_bytes("test_input\\1.jpg")
data = json.dumps({'instances': [{"b64": b64, "pretrained": 'sketch_multi'}]})
# print(data)

headers = {"content-type": "application/json"}
json_response = requests.post('http://172.18.69.27:5001/v1/models/face_edit_generate/versions/1/',
                              data=data, headers=headers)

predictions = json.loads(json_response.text)['predictions']
img = np.array(predictions)
plt.imshow(img)
plt.show()
