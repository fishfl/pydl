# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md
# https://drive.google.com/u/0/uc?export=download&confirm=eqPD&id=1o6ijA3PkcewZvwJJ73dJ0fxhndn0nnh7

import torch

import e4e_predict_holder as e4e_h
import model_holder as mh

from PIL import Image
import base64
from flask import Flask, jsonify, request, redirect
from io import BytesIO
from util import align_face2


app = Flask(__name__)
error_str = '{"error": %s}'
rst_func = lambda rst: {"predictions": rst}


def read_img(req):
    try:
        data = req.json
        instance = data['instances'][0]  # 默认只接收第一张图片
        b64 = instance['b64']
        pretrained = instance['pretrained']
        # read the image with b64 str and convert to np
        b = b64.encode()
        i_b = base64.b64decode(b)
        image_data = BytesIO(i_b)
        img = Image.open(image_data)
        return img, pretrained, None
    except BaseException as e:
        error = error_str % str(e)
        return None, None, error


@app.route('/v1/models/face_edit_generate/versions/1/', methods=['POST'])
def predict():
    img, pretrained, error = read_img(request)
    if (img is not None) and (pretrained is not None):
        aligned_face = align_face2(img)
        my_w = e4e_h.projection(aligned_face).unsqueeze(0)
        generator = mh.get_or_create_pretrained(pretrained)
        stylized_face = generator(my_w, input_is_latent=True)

        stylized_face = 1 + stylized_face
        stylized_face /= 2
        stylized_face = stylized_face[0]
        stylized_face = 255 * torch.clip(stylized_face, min=0, max=1)
        stylized_face = stylized_face.byte()
        stylized_face = stylized_face.permute(1, 2, 0).detach().numpy()

        return rst_func(stylized_face.tolist())
    else:
        return error


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False)