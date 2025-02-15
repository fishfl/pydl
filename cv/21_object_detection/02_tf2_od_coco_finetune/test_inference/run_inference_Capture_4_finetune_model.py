# !pip install -U --pre tensorflow=="2.*"
# !pip install tf_slim
# !pip install pycocotools

import pathlib

# if "models" in pathlib.Path.cwd().parts:
#   while "models" in pathlib.Path.cwd().parts:
#     os.chdir('..')
# elif not pathlib.Path('models').exists():
#   !git clone --depth 1 https://github.com/tensorflow/models

# %%bash
# cd models/research/
# protoc object_detection/protos/*.proto --python_out=.
#
# %%bash
# cd models/research
# pip install .

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def load_model():
  model = tf.saved_model.load("../test_inference/inference_graph/saved_model")
  return model

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '../training/cat_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('../test_inference/test_images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpeg")))

detection_model = load_model()

print(detection_model.signatures['serving_default'].inputs)
print(detection_model.signatures['serving_default'].output_dtypes)
print(detection_model.signatures['serving_default'].output_shapes)
print(detection_model.signatures['serving_default'].output)

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

  return output_dict

def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
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

for image_path in TEST_IMAGE_PATHS:
  show_inference(detection_model, image_path)


#
#
# import cv2
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, image_np = cap.read()
#     image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
#
#     output_dict = run_inference_for_single_image(detection_model, image_np)
#     # Visualization of the results of a detection.
#     vis_util.visualize_boxes_and_labels_on_image_array(
#         image_np,
#         output_dict['detection_boxes'],
#         output_dict['detection_classes'],
#         output_dict['detection_scores'],
#         category_index,
#         use_normalized_coordinates=True,
#         line_thickness=8)
#
#     cv2.imshow('object detection', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cap.release()
#         cv2.destroyAllWindows()
#         break
