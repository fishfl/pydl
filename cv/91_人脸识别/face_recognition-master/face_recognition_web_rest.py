import base64

import face_recognition
from flask import Flask, jsonify, request, redirect
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)
error_str = '{"error": %s}'
rst_func = lambda rst: {"predictions": rst}


def read_img(req):
    try:
        data = req.json
        instance = data['instances'][0]  #默认只接收第一张图片
        b64 = instance['b64']
        # read the image with b64 str and convert to np
        b = b64.encode()
        i_b = base64.b64decode(b)
        image_data = BytesIO(i_b)
        img = Image.open(image_data)
        np_arr = np.array(img)
        return np_arr, None
    except BaseException as e:
        error = error_str % str(e)
        return None, error


@app.route('/v1/models/fr_face_locations/versions/1/', methods=['POST'])
def face_locations():
    img, error = read_img(request)
    if img is not None:
        rst = face_recognition.face_locations(img)
        return rst_func(rst)
    else:
        return error

@app.route('/v1/models/fr_face_landmarks/versions/1/', methods=['POST'])
def face_landmarks():
    img, error = read_img(request)
    if img is not None:
        rst = face_recognition.face_landmarks(img)
        return rst_func(rst)
    else:
        return error

@app.route('/v1/models/fr_face_encodings/versions/1/', methods=['POST'])
def face_encodings():
    img, error = read_img(request)
    if img is not None:
        rst = face_recognition.face_encodings(img)
        return rst_func(str(rst))
    else:
        return error


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False)
