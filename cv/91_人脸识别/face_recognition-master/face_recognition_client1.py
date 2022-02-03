import json
from PIL import Image
import numpy as np
import requests
import base64

def image_bytes(file):  #--same with read_gfile
    with open(file, "rb") as imageFile:
        ib = base64.b64encode(imageFile.read()).decode()
        return ib

def read_byte(file):
    i = Image.open(file)
    img = np.array(i)
    return img

b64 = image_bytes("D:\\1.jpg")
image = read_byte("D:\\1.jpg")

data = json.dumps({'instances': [{"b64": b64}]})
# print(data)

headers = {"content-type": "application/json"}
json_response = requests.post('http://172.27.152.172:5001/v1/models/fr_face_locations/versions/1/',
        data=data, headers=headers)

predictions = json.loads(json_response.text)['predictions']
print(predictions)

for face_location in predictions:
    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()


