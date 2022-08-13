import json
import numpy as np
import requests
import base64
import matplotlib.pyplot as plt


def image_bytes(file):
    with open(file, "rb") as imageFile:
        ib = base64.b64encode(imageFile.read()).decode()
        return ib


b64 = image_bytes("d:\\1.jpg")
data = json.dumps({'instances': [{"b64": b64, "pretrained": 'arcane_caitlyn'}]})
print(data)

headers = {"content-type": "application/json",
           "user_id": "746478570813272064",
           "token": "f0d77297-9e10-4eb9-853a-2fdb420a368c",
           "model_name": "face-edit-generate-offical"}

json_response = requests.post('http://aimodelmarket.cn:8080/v1/models/face_edit_generate/versions/1/',
                              data=data, headers=headers)

predictions = json.loads(json_response.text)['predictions']
img = np.array(predictions)
plt.imshow(img)
plt.show()
