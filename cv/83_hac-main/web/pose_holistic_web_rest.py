import sys
sys.path.insert(0, "/root/hac/pyhac/")
from pyhac.tracker import HolisticTracker
from PIL import Image
import numpy as np
import base64
from flask import Flask, jsonify, request, redirect
from io import BytesIO

app = Flask(__name__)
error_str = '{"error": %s}'
rst_func = lambda rst: {"predictions": rst}


def read_img(req):
    try:
        data = req.json
        instance = data['instances'][0]  # 默认只接收第一张图片
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

@app.route('/v1/models/pe_holistic_keypoints/versions/1/', methods=['POST'])
def pe_holistic_keypoints():
    img, error = read_img(request)
    if img is not None:
        tracker = HolisticTracker()
        results = tracker.holistic.process(img)
        keypoints_dict = tracker.pred_to_dict(results, 0)
        rst = keypoints_dict
        return rst_func(rst)
    else:
        return error

@app.route('/v1/models/pe_holistic_image/versions/1/', methods=['POST'])
def pe_holistic_image():
    img, error = read_img(request)
    if img is not None:
        tracker = HolisticTracker()
        tracker.results = tracker.holistic.process(img)
        img_rst = tracker.draw_landmarks(img)
        return rst_func(img_rst.tolist())
    else:
        return error

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False)


