
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
tf = tf.compat.v1

export_dir = 'D:\\models\\ad_verify\\out\\1'
graph_pb = 'D:\\models\\ad_verify\\output_graph.pb'

builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

with tf.gfile.GFile(graph_pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    # for n in graph_def.node:
    #     print(n.name)
sigs = {}

with tf.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graph_def, name="")
    g = tf.get_default_graph()

    inp = g.get_tensor_by_name("DecodeJpeg/contents:0")
    out = g.get_tensor_by_name("final_result:0")

    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def({"in": inp}, {"out": out})

    builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING],signature_def_map = sigs)

builder.save()
