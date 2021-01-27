#!/usr/bin/env python3
import sys
import json
import argparse
import cv2
import time
import numpy as np

from google.protobuf.json_format import MessageToJson

import tensorflow as tf
#from tensorflow.python.tools import freeze_graph

def parse_arguments(argv):
    parser = argparse.ArgumentParser(argv[0])
    parser.add_argument('sv_file', type=str)

    return parser.parse_args(argv[1:])

def get_model_info(tf_model_filename):
    ops = {}
    with tf.Session() as sess:
        with tf.gfile.GFile(tf_model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            _ = tf.import_graph_def(graph_def)
            for op in tf.get_default_graph().get_operations():
                ops[op.name] = [str(output) for output in op.outputs]
        writer = tf.summary.FileWriter('./logs')
        writer.add_graph(sess.graph)
        writer.flush()
        writer.close()

    j = MessageToJson(graph_def)
    with open('./model.json', 'w') as f:
        f.write(j)

def load_model(model_dir):

    model = tf.saved_model.load(str(model_dir))
    return model

def inference_model(model):

    # load a sample image
    img = cv2.imread('resources/images/img_99.png')
    img = cv2.resize(img, (300, 300))
    img = img.reshape(1, *img.shape)  # 1x300x300x3
    image = np.asarray(img)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(img)

    # Run inference
    infer_ms_l = []
    for i in range(0, 1000): # run 1000 iterations
        model_fn = model.signatures['serving_default']
        start_s = time.time()
        output_dict = model_fn(input_tensor)
        infer_ms = (time.time() - start_s) * 1e3
        infer_ms_l.append(infer_ms)
        print(f'infer_ms: {infer_ms} ms')

#    print(f'output: {output_dict}')

    avg_infer_ms = sum(infer_ms_l) / len(infer_ms_l)
    print(f'average inference time: {avg_infer_ms} ms')

def main():
    args = parse_arguments(sys.argv)

    # get_model_info(args.pb_file)

    # load a model
    m = load_model(args.sv_file)
    # inference
    inference_model(m)


if __name__ == '__main__':
    main()
