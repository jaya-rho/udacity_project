'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2

from openvino.inference_engine import IENetwork, IECore

from logger import Logger
logger = Logger.get_logger('logger')

class Model_FacialLandmarksDetection:
    """
    Class for the Facial Landmarks Detection Model.
    https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html
    """
    def __init__(self, model_name, device='CPU', extensions=None):
        """
        class initialization
        """
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
        """
        load the model to the specified device by the user
        """
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
        """
        This method is meant for running predictions on the input image.
        inference with an asynchronous request
        """
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

    def preprocess_input(self, image):
        '''
        preprocess before feeding the data into the model for inference
        # shape: [1x3x48x48] - An input image in the format [BxCxHxW]
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
        preprocess before feeding the output of this model to the next model
        # shape: [1, 10, 1, 1]
        # a row-vector of 10 floating point values for five landmarks coordinates in the form (x0, y0, x1, y1, ..., x4, y4)
        '''
        detected_bboxes = []
        # output: [[[[ e1, e2, e3, .. , e7 ]]]]
        left_eye_x = outputs[0][0][0][0]
        left_eye_y = outputs[0][1][0][0]
        right_eye_x = outputs[0][2][0][0]
        right_eye_y = outputs[0][3][0][0]

        logger.debug(f'[raw output]')
        logger.debug(f'  left_eye_x: {left_eye_x}')
        logger.debug(f'  left_eye_y: {left_eye_y}')
        logger.debug(f'  right_eye_x: {right_eye_x}')
        logger.debug(f'  right_eye_y: {right_eye_y}')
                
        return [left_eye_x, left_eye_y, right_eye_x, right_eye_y]

