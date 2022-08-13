import json
import requests
import base64


def image_bytes(file):
    with open(file, "rb") as imageFile:
        ib = base64.b64encode(imageFile.read()).decode()
        return ib


b64 = image_bytes("d:\\2.jpg")

data = json.dumps({'instances': [{"b64": b64}]})
# print(data)

headers = {"content-type": "application/json",
           "user_id": "746478570813272064",
           "token": "f0d77297-9e10-4eb9-853a-2fdb420a368c",
           "model_name": "face-recognition-offical"
           }
json_response = requests.post('http://aimodelmarket.cn:8080/v1/models/fr_face_encodings/versions/1/',
        data=data, headers=headers)

predictions = json.loads(json_response.text)['predictions']
print(predictions)
