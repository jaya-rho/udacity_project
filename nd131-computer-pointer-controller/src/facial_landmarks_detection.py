'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2

from openvino.inference_engine import IENetwork, IECore
from base_model import Model

from logger import Logger
logger = Logger.get_logger('logger')

class Model_FacialLandmarksDetection(Model):
    """
    Class for the Facial Landmarks Detection Model.
    https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html
    """
    def __init__(self, model_name, device='CPU', extensions=None):
        """
        class initialization
        """
        super().__init__(model_name, device, extensions) # initialize the base class

    def predict(self, cropped_face_frame):
        """
        This method is meant for running predictions on the input image.
        inference with an asynchronous request
        """
#        self.exec_network_.start_async(request_id=request_id,inputs={self.input_blob_: image})
        preproc_frame = self.preprocess_input(cropped_face_frame)
        self.exec_network_.infer(inputs={self.input_blob_: preproc_frame})
        infer_res = self.get_output()
        eyes_coord, cropped_left_eye, cropped_right_eye = self.preprocess_output(infer_res, cropped_face_frame)

        return eyes_coord, cropped_left_eye, cropped_right_eye

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

    def preprocess_output(self, outputs, cropped_face_frame):
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

        cropped_face_h = cropped_face_frame.shape[0]
        cropped_face_w = cropped_face_frame.shape[1]

        left_eye_x_i = int(cropped_face_w * left_eye_x)
        left_eye_y_i = int(cropped_face_h * left_eye_y)
        right_eye_x_i = int(cropped_face_w * right_eye_x)
        right_eye_y_i = int(cropped_face_h * right_eye_y)

        eyes_coord = [left_eye_x_i, left_eye_y_i, right_eye_x_i, right_eye_y_i]

        logger.debug(f'[valid eye bbox on frame]')
        logger.debug(f'  left_eye x: {left_eye_x_i}')
        logger.debug(f'  left_eye y: {left_eye_y_i}')
        logger.debug(f'  right_eye x: {right_eye_x_i}')
        logger.debug(f'  right_eye y: {right_eye_y_i}')

        cropped_left_eye = cropped_face_frame[(left_eye_y_i - 20):(left_eye_y_i + 20), (left_eye_x_i - 20):(left_eye_x_i + 20)]
        cropped_right_eye = cropped_face_frame[(right_eye_y_i - 20):(right_eye_y_i + 20), (right_eye_x_i - 20):(right_eye_x_i + 20)]

        return eyes_coord, cropped_left_eye, cropped_right_eye

