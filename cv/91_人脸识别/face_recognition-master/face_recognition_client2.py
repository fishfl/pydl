import json
from PIL import Image, ImageDraw
import numpy as np
import requests
import base64


def image_bytes(file):
    with open(file, "rb") as imageFile:
        ib = base64.b64encode(imageFile.read()).decode()
        return ib


b64 = image_bytes("d:\\2.jpg")
data = json.dumps({'instances': [{"b64": b64}]})

headers = {"content-type": "application/json",
           "user_id": "746478570813272064",
           "token": "f0d77297-9e10-4eb9-853a-2fdb420a368c",
           "model_name": "face-recognition-offical"
           }
json_response = requests.post('http://aimodelmarket.cn:8080/v1/models/fr_face_landmarks/versions/1/',
        data=data, headers=headers)

predictions = json.loads(json_response.text)['predictions']
print(predictions)




def read_byte(file):
    i = Image.open(file)
    img = np.array(i)
    return img
image = read_byte("D:\\2.jpg")


pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)

for face_landmarks in predictions:

    # Print the location of each facial feature in this image
    for facial_feature in face_landmarks.keys():
        print("{} : {}".format(facial_feature, face_landmarks[facial_feature]))

    # Let's trace out each facial feature in the image with a line!
    for facial_feature in face_landmarks.keys():
        tl = [(f[0], f[1]) for f in face_landmarks[facial_feature]]
        d.line(tl, width=5)

# Show the picture
pil_image.show()

