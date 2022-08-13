import json
import requests
import base64

def image_bytes(file):  #--same with read_gfile
    with open(file, "rb") as imageFile:
        ib = base64.b64encode(imageFile.read()).decode()
        return ib

i = image_bytes("d:\\2.jpg")

data = '{"signature_name": "serving_default", "inputs": {"in" : {"b64": "%s"}}}' % i
print(data)
headers = {"content-type": "application/json",
           "user_id": "746478570813272064",
           "token": "f0d77297-9e10-4eb9-853a-2fdb420a368c",
           "model_name": "violence-verify-offical"
           }
json_response = requests.post('http://aimodelmarket.cn:8080/v1/models/violence_verify:predict',
        data=data, headers=headers)

predictions = json.loads(json_response.text)
print(predictions)
