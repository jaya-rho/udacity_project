#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.infer_engine_ = None
        self.network_ = None
        self.exec_network_ = None
        self.input_blob_ = []
        self.output_blob_ = []


    def load_model(self, xml_model, device='CPU', cpu_ext=None):
        ### TODO: Load the model ###
#        print('* load model *')
        model_dir_path = os.path.splitext(xml_model)[0]
        bin_model = model_dir_path + '.bin'
#        print(' - xml model: %s' % (xml_model))
#        print(' - bin model: %s' % (bin_model))

        # Initialize the Inference Engine
        self.infer_engine_ = IECore()

        ### TODO: Add any necessary extensions ###
        if cpu_ext and "CPU" in device:
            self.infer_engine_.add_extension(cpu_ext, device)

        # Read the IR as a IENetwork
        self.network_ = self.infer_engine_.read_network(model=xml_model, weights=bin_model)

        ### TODO: Check for supported layers ###
        layers_map = self.infer_engine_.query_network(self.network_, "CPU")

#        print('* layers info *')
        layer_num = 0
        unsupport_layers = []
        for layer, hw in layers_map.items():
            if not hw in ['CPU']:
#                print(' [U] #%d: %s [%s]' % (layer_num, layer, hw))
                unsupport_layers.append(layer)
            else:
#                print(' [S] #%d: %s [%s]' % (layer_num, layer, hw))
                pass
            layer_num += 1

#        print(' - unsupported layers: %s' % unsupport_layers)

        ### TODO: Return the loaded inference plugin ###
        self.exec_network_ = self.infer_engine_.load_network(self.network_, device)

        ### Note: You may need to update the function parameters. ###
        self.input_blob_ = next(iter(self.network_.inputs))
        self.output_blob_ = next(iter(self.network_.outputs))

#        print('* blob info *')
#        print(' - input : %s' % self.input_blob_)
#        print(' - output: %s' % self.output_blob_)

        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.network_.inputs[self.input_blob_].shape

    def exec_net(self, image):
        ### TODO: Start an asynchronous request ###
        self.exec_network_.start_async(request_id=0,inputs={self.input_blob_: image})

        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        infer_status = self.exec_network_.requests[0].wait(-1)
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return infer_status

    def get_output(self):
        ### TODO: Extract and return the output results
        infer_output = self.exec_network_.requests[0].outputs[self.output_blob_]
#        print('  - extracting DNN output from blob (%s)' % self.output_blob_)
        # DNN output  [1, 1, N, 7], [image_id, label, conf, x_min, y_min, x_max, y_max]
#        print('  - output shape: {}'.format(infer_output.shape))

        ### Note: You may need to update the function parameters. ###
        return infer_output

