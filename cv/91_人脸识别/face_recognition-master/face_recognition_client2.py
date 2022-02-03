import json
from PIL import Image, ImageDraw
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
json_response = requests.post('http://172.27.152.172:5001/v1/models/fr_face_landmarks/versions/1/',
        data=data, headers=headers)

predictions = json.loads(json_response.text)['predictions']
print(predictions)

pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)

for face_landmarks in predictions:

    # Print the location of each facial feature in this image
    for facial_feature in face_landmarks.keys():
        print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

    # Let's trace out each facial feature in the image with a line!
    for facial_feature in face_landmarks.keys():
        tl = [(f[0], f[1]) for f in face_landmarks[facial_feature]]
        d.line(tl, width=5)

# Show the picture
pil_image.show()

