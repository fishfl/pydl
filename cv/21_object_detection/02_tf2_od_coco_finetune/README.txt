
https://gilberttanner.com/blog/tensorflow-object-detection-with-tensorflow-2-creating-a-custom-model

# train dataset
python xml_to_csv.py

# eval dataset
python xml_to_csv.py

python generate_tfrecord.py --csv_input=img_csv/cat_train_labels.csv --images_input=images/train --output_path=tfrecord/train/cat.record --label_map_path=training/cat_label_map.pbtxt

python generate_tfrecord.py --csv_input=img_csv/cat_eval_labels.csv --images_input=images/eval --output_path=tfrecord/eval/cat.record --label_map_path=training/cat_label_map.pbtxt

#https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
#下载的模型要和configs/tf2/下的配置文件相对应，将配置文件copy到training下
#如：
#http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz



#Change fine_tune_checkpoint_type to detection


python ../object_detection/model_main_tf2.py --model_dir=training --pipeline_config_path=training/ssd_efficientdet_d0_512x512_coco17_tpu-8.config

python ../object_detection/exporter_main_v2.py --trained_checkpoint_dir=training --pipeline_config_path=training/ssd_efficientdet_d0_512x512_coco17_tpu-8.config --output_directory inference_graph