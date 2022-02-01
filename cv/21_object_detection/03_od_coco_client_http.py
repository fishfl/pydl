import json
from PIL import Image
import numpy as np
import requests
from matplotlib import pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def read_byte(file):
    i = Image.open(file)
    img = np.array(i)
    jpeg_rgb = np.expand_dims(img, 0).tolist()
    return img,jpeg_rgb

image_np, i = read_byte("D:\\1.jpg")

data = json.dumps({'instances': i})
# print(data)

headers = {"content-type": "application/json"}
json_response = requests.post('http://172.27.144.21:8501/v1/models/od_coco:predict',
        data=data, headers=headers)


predictions = json.loads(json_response.text)['predictions']
print(predictions)

output_dict = predictions[0]
# All outputs are batches tensors.
# Convert to numpy arrays, and take index [0] to remove the batch dimension.
# We're only interested in the first num_detections.
num_detections = int(output_dict.pop('num_detections'))
output_dict = {key: np.array(value) for key, value in output_dict.items()}
output_dict['num_detections'] = num_detections
# detection_classes should be ints.
output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Visualization of the results of a detection.
vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      line_thickness=8)

import matplotlib;matplotlib.use('TkAgg')

plt.figure(figsize=(12,16))
plt.imshow(image_np)
plt.show()
