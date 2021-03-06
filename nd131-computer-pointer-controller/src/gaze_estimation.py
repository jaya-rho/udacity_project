'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import math

from openvino.inference_engine import IENetwork, IECore

from logger import Logger
logger = Logger.get_logger('logger')

class Model_GazeEstimation:
    '''
    Class for the Gaze Estimation Model.
    https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html
    '''
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
        self.load_model()

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

        self.input_blob_ = list(self.network_.inputs.keys())
        self.output_blob_ = next(iter(self.network_.outputs))

        logger.debug('* blob info *')
        logger.debug(' - input : %s' % self.input_blob_)
        logger.debug(' - output: %s' % self.output_blob_)

    def get_input_shape(self):
        """ Return the shape of the input layer """
        input_shape_d = {}
        for i in self.input_blob_:
            input_shape_d[i] = self.network_.inputs[i].shape

        return input_shape_d

    def predict(self, cropped_left_eye, cropped_right_eye, hp_angle, request_id=0):
        """
        This method is meant for running predictions on the input image.
        inference with an asynchronous request

        # input #1: left_eye_image and the shape [1x3x60x60] (BxCxHxW)
        # input #2: right_eye_image and the shape [1x3x60x60]
        # input #3: head_pose_angles and the shape [1x3] (BxC)
        """

        preproc_left_eye = self.preprocess_input(cropped_left_eye, "left_eye_image")
        preproc_right_eye = self.preprocess_input(cropped_right_eye, "right_eye_image")
        self.exec_network_.infer(inputs={
            "left_eye_image": preproc_left_eye,
            "right_eye_image": preproc_right_eye,
            "head_pose_angles": hp_angle})
#        self.exec_network_.start_async(request_id=request_id,inputs={
#            "left_eye_image":le_image,
#            "right_eye_image":re_image,
#            "head_pose_angles":hp_angle
#            })

        infer_result = self.get_output()
        mouse_coords, gaze_vec = self.preprocess_output(infer_result, hp_angle)

        return mouse_coords, gaze_vec

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

    def preprocess_input(self, image, input_name):
        '''
        preprocess before feeding the data into the model for inference
        # input #1: left_eye_image and the shape [1x3x60x60] (BxCxHxW)
        # input #2: right_eye_image and the shape [1x3x60x60]
        '''
        input_shape_d = self.get_input_shape()
        print(f' *** {input_name} shape: {input_shape_d[input_name]}')
        [n, c, h, w] = input_shape_d[input_name]

        converted_frame = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)

        # convert HWC into CHW
        converted_frame = converted_frame.transpose((2, 0, 1))
        # convert shape NCHW
        converted_frame = converted_frame.reshape((n, c, h, w))

        logger.debug(f'input [{image.shape}] -> converted_input [{converted_frame.shape}]')
        return converted_frame

    def preprocess_output(self, outputs, hp_angle):
        '''
        preprocess before feeding the output of this model to the next model
        # output: gaze_vector with the shape: [1, 3]
        '''
        print(f'outputs: {outputs}')
        print(f'head pose angle: {hp_angle}')
        gaze_vec = outputs[0]
        angle_r_fc = hp_angle[2]
        cosine = math.cos(angle_r_fc * math.pi / 180.0)
        sine = math.sin(angle_r_fc * math.pi / 180.0)

        x_val = gaze_vec[0] * cosine + gaze_vec[1] * sine
        y_val = -gaze_vec[0] * sine + gaze_vec[1] * cosine
                
        return (x_val, y_val), gaze_vec

