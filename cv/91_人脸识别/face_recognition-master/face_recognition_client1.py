import json
from PIL import Image
import numpy as np
import requests
import base64


def image_bytes(file):
    with open(file, "rb") as imageFile:
        ib = base64.b64encode(imageFile.read()).decode()
        return ib


b64 = image_bytes("d:\\1.jpg")
data = json.dumps({'instances': [{"b64": b64}]})
print(data)

headers = {"content-type": "application/json",
           "user_id": "746478570813272064",
           "token": "f0d77297-9e10-4eb9-853a-2fdb420a368c",
           "model_name": "face-recognition-offical"
           }
json_response = requests.post('http://aimodelmarket.cn:8080/v1/models/fr_face_locations/versions/1/',
        data=data, headers=headers)

predictions = json.loads(json_response.text)['predictions']
print(predictions)






def read_byte(file):
    i = Image.open(file)
    img = np.array(i)
    return img

image = read_byte("D:\\1.jpg")

for face_location in predictions:
    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()


