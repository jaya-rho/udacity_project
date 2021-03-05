'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2

from openvino.inference_engine import IENetwork, IECore

from logger import Logger
logger = Logger.get_logger('logger')

class Model_FaceDetection:
    '''
    Class for the Face Detection Model.
    https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_0001_description_face_detection_adas_0001.html
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.bin_model_ = model_name + '.bin'
        self.xml_model_ = model_name + '.xml'
        self.device_ = device
        self.extensions_ = extensions
        self.infer_engine_ = None
        self.network_ = None
        self.exec_network_ = None
        self.input_blob_ = []
        self.output_blob_ = []

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''

        # Initialize the Inference Engine
        self.infer_engine_ = IECore()

        if self.extensions_ and "CPU" in self.device_:
            self.infer_engine_.add_extension(self.extensions_, self.device_)

        # Read the IR as a IENetwork
        self.network_ = self.infer_engine_.read_network(model=self.xml_model_, weights=self.bin_model_)
        # Return the loaded inference plugin ###
        self.exec_network_ = self.infer_engine_.load_network(self.network_, self.device_)

        self.input_blob_ = next(iter(self.network_.inputs))
        self.output_blob_ = next(iter(self.network_.outputs))

        logger.debug('* blob info *')
        logger.debug(' - input : %s' % self.input_blob_)
        logger.debug(' - output: %s' % self.output_blob_)

    def get_input_shape(self):
        """ Return the shape of the input layer """
        return self.network_.inputs[self.input_blob_].shape

    def predict(self, image, request_id=0):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        self.exec_network_.start_async(request_id=request_id,inputs={self.input_blob_: image})
        return

    def wait(self):
        """ Wait for the request to be complete """
        infer_status = self.exec_network_.requests[0].wait(-1)
        return infer_status

    def get_output(self):
        """ Extract and return the output results """
        infer_output = self.exec_network_.requests[0].outputs[self.output_blob_]
        logger.debug('  - extracting DNN output from blob (%s)' % self.output_blob_)
        logger.debug('  - output shape: {}'.format(infer_output.shape))
        return infer_output

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        # shape: [1x3x384x672] - An input image in the format [BxCxHxW],
        '''
        [n, c, h, w] = self.get_input_shape()

        converted_frame = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)

        # convert HWC into CHW
        converted_frame = converted_frame.transpose((2, 0, 1))
        # convert shape NCHW
        converted_frame = converted_frame.reshape((n, c, h, w))

        logger.debug(f'input [{image.shape}] -> converted_input [{converted_frame.shape}]')
        return converted_frame

    def preprocess_output(self, outputs, conf_thres):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        # [1, 1, N, 7] where N is the number of detected bounding boxes
        # [image_id, label, conf, x_min, y_min, x_max, y_max]
        '''
        detected_bboxes = []
        # output: [[[[ e1, e2, e3, .. , e7 ]]]]
        for bbox in outputs[0][0]:
            confidence = bbox[2]
            if confidence > conf_thres:
                valid_bbox = (bbox[3], bbox[4], bbox[5], bbox[6])
                logger.debug(f'append bbox [{bbox[3]},{bbox[4]},{bbox[5]},{bbox[6]}]')
                detected_bboxes.append(valid_bbox)

        top_detected_bbox = detected_bboxes[0]
        #TODO: add assertion
                
        return top_detected_bbox

