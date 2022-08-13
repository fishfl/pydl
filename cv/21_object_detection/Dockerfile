FROM tensorflow/serving:2.4.3-devel

WORKDIR /

RUN mkdir -p /models/od_cat/1

ADD ./02_tf2_od_coco_finetune/test_inference/inference_graph/saved_model  /models/od_cat/1

CMD tensorflow_model_server --rest_api_port=8501 --port=8500 --model_name=od_cat --model_base_path=/models/od_cat

