"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import io
import sys
import time
import socket
import json
import cv2
import time

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
from sys import platform

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


# Get correct CPU extension
if platform == "linux" or platform == "linux2":
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
    CODEC = 0x00000021
elif platform == "darwin":
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib"
 #   CODEC = cv2.VideoWriter_fourcc('M','J','P','G')  # it does not work on MAC
    CODEC = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
else:
    print("Unsupported OS.")
    exit(1)


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    """ Connect to the MQTT client """
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def extract_valid_object(infer_output):
    """
    Extract the valid object (person) from the DNN inference output

    :param image: an input image
    :return valid_obj: a list consisting of the valid objects' info
    """

    valid_obj = []

    for b_i, b in enumerate(infer_output[:,:,:,:]):
        for c_i, c in enumerate(b[:,:,:]):
            for h_i, h in enumerate(c[:,:]):
                obj_dict = {}
                obj_dict['image_id'] = h[0]
                obj_dict['label'] = h[1]
                obj_dict['conf'] = h[2]
                obj_dict['x_min'] = h[3]
                obj_dict['y_min'] = h[4]
                obj_dict['x_max'] = h[5]
                obj_dict['y_max'] = h[6]

                # append only the valid object
                if not all(v == 0 for v in obj_dict.values()):
                    valid_obj.append(obj_dict)

#    print('res: {}'.format(valid_obj))

    return valid_obj

def draw_boundingbox(image, infer_output, image_width, image_height, conf_thresh):
    """
    compute the coordinate of bounding box from DNN inference output
    filter the valid object by the confidence threshold

    :param image: an input image
    :param infer_output: the inference results
    :param image_width: a width of an input image
    :param image_height: a height of an input image
    :param conf_thresh: a confidence threshold
    :return out_image: an output image which a bounding box is drawn
    :return valid_obj_num: the number of valid detected object
    """

    out_image = image.copy()
#    print('  - input image: [width] %d, [height] %d' % (image.shape[1], image.shape[0]))

    def check_valid_range(val, max_val):
        """ check the coordinate of bbox is inside of an image"""
        if val < 0:
            val = 0
        elif val > max_val:
            val = max_val
        else:
            pass
        return val

    valid_obj_num = 0
    valid_obj_bbox = []

    for obj_info in infer_output:
        conf = obj_info['conf']
        # filter by the confidence
        if conf >= conf_thresh:
            # calculate bbox coordinate
            xmin = int(obj_info['x_min'] * image_width)
            ymin = int(obj_info['y_min'] * image_height)
            xmax = int(obj_info['x_max'] * image_width)
            ymax = int(obj_info['y_max'] * image_height)

            # round up into valid range
            xmin = check_valid_range(xmin, image_width)
            ymin = check_valid_range(ymin, image_height)
            xmax = check_valid_range(xmax, image_width)
            ymax = check_valid_range(ymax, image_height)

            # draw bbox
            cv2.rectangle(out_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

            valid_obj_num += 1
            valid_obj_bbox.append((xmin, ymin, xmax, ymax))
#            print('  - draw bbox [%d, %d, %d, %d] confidence: %f' % (xmin,ymin,xmax,ymax,conf))

    # assert if one more people are detected per frame
    if valid_obj_num > 1:
        assert False, 'people counter > 1 in one frame'

    return out_image, valid_obj_num


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialize the network
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    # Load the model through `infer_network`
    infer_network.load_model(args.model, args.device, args.cpu_extension)

    # Handle the input stream
    # expand the tilde
    input_fpath = os.path.expanduser(args.input)
    f_name, f_extn = os.path.splitext(input_fpath)

    is_image_input = False

    # add the file extensions as you like
    if f_extn in ['.mp4', '.avi', '.mpeg']:
        pass
    elif f_extn in ['.png', '.jpg', 'jpeg']:
        is_image_input = True
    else:
        assert False, f'unsupported input data extension: {f_extn}'

    # Get and open video capture
    cap = cv2.VideoCapture(input_fpath)
    cap.open(input_fpath)
    # [1, 3, 320, 544] (BCHW)
    net_input_dims = infer_network.get_input_shape()
#    print('* DNN input dims: {}'.format(net_input_dims))

    width = int(cap.get(3))
    height = int(cap.get(4))
    # * Video dims: [height:432, width:768]
#    print('* Video dims: [height:{}, width:{}]'.format(height, width))

#    print('platform: {}'.format(platform))
    out_video = cv2.VideoWriter('out_result.mp4', CODEC, 30, (width, height))

    # Loop until stream is over
    frame_num = 0
    last_valid_pers_num = 0
    total_valid_pers_num = 0
    duration_time_sec = 0
    mis_detect_cnt = [False]*3

    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Pre-processing the image
        # cv2.resize(src, dsize=(width, height))
        p_frame = cv2.resize(frame, (net_input_dims[3], net_input_dims[2]))
        p_frame = p_frame.transpose((2,0,1))
        # reshape (3, 320, 544) to (1, 3, 320, 544)
        p_frame = p_frame.reshape(1, *p_frame.shape)
#        print('+ frame %d' % (frame_num))
#        print('  - shape: {}'.format(p_frame.shape))

        # Start asynchronous inference for specified request
        infer_start = time.time()
        infer_network.exec_net(p_frame)

        # Wait for the result
        if infer_network.wait() == 0: # when the inference per frame finishes
            infer_stop = time.time()
            infer_time_ms = (infer_stop-infer_start) * 1e3

            # Get the results of the inference request
            infer_result = infer_network.get_output()

            # Filter the valid object
            valid_object = extract_valid_object(infer_result)

            # draw bounding box of detected person on the image
            out_frame, valid_pers_num = draw_boundingbox(frame, valid_object, width, height, prob_threshold)

            def add_text_on_image(image, insert_text=None, loc=(10,10), tsize=0.4, tcolr=(209, 130, 0, 255), tbold=1):
                # add a text
                cv2.putText(image, insert_text, loc, cv2.FONT_HERSHEY_SIMPLEX, tsize, tcolr, tbold)
#                print('  - [add the text on image] %s' % (insert_text))
                return

#            print('  - total number of people: %d' % total_valid_pers_num)

            if valid_pers_num > last_valid_pers_num:
                # the number of detected people increases
                emerge_time = time.time()
                total_valid_pers_num += (valid_pers_num - last_valid_pers_num)
            elif valid_pers_num < last_valid_pers_num:
                # the number of detected people decreases
                assert emerge_time >= 0, 'a person emerged time is not valid ({})'.format(emerge_time)
                duration_time_sec = time.time() - emerge_time
            else:
                # the number of detected people not changed, do nothing
                pass

#            print('  - result: {}'.format(mis_detect_cnt))
            insert_text = 'duration time: %d sec' % (duration_time_sec)
            add_text_on_image(out_frame, insert_text, (10,60))

            insert_text = 'total count of people: %d' % (total_valid_pers_num)
            add_text_on_image(out_frame, insert_text, (10,40))
            last_valid_pers_num = valid_pers_num

            # add inference time on the image
            insert_text = "inference time(without post-process): %.2fms" % (infer_time_ms)
            add_text_on_image(out_frame, insert_text, (10,20))

            if is_image_input:
                path = '.'
                f_name = f'output_{frame_num}{f_extn}'
                cv2.imwrite(os.path.join(path, f_name), out_frame)
            else:
                # write into a movie
                out_video.write(out_frame)

            # Send current_count, total_count and duration to the MQTT server ###
            # Topic "person": keys of "count" and "total"
            client.publish("person", json.dumps({"count": valid_pers_num}))
            client.publish("person", json.dumps({"total": total_valid_pers_num}))

            # Topic "person/duration": key of "duration"
            client.publish("person/duration", json.dumps({"duration": duration_time_sec}))

        # Send the frame to the FFMPEG server
        sys.stdout.buffer.write(out_frame)
        sys.stdout.flush()

        # Break if escape key pressed
        if key_pressed == 27:
            break

        # count up the frame number
        frame_num += 1

    # Release the capture and destroy any OpenCV windows
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

    # close the MTQQ server connection
    client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()

    # Connect to the MQTT server
    client = connect_mqtt()

    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
