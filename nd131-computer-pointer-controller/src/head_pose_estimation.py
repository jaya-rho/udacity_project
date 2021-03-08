'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2

from openvino.inference_engine import IENetwork, IECore
from base_model import Model

from logger import Logger
logger = Logger.get_logger('logger')

class Model_HeadPoseEstimation(Model):
    '''
    Class for the Head Post Estimation Model.
    https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        """
        class initialization
        """
        super().__init__(model_name, device, extensions) # initialize the base class

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
        self.output_blob_ = list(self.network_.outputs.keys())

        logger.debug('* blob info *')
        logger.debug(' - input : %s' % self.input_blob_)
        logger.debug(' - output: %s' % self.output_blob_)

    def predict(self, frame):
        """
        This method is meant for running predictions on the input image.
        inference with an asynchronous request
        """
        preproc_frame = self.preprocess_input(frame)
#        self.exec_network_.start_async(request_id=request_id,inputs={self.input_blob_: image})
        self.exec_network_.infer(inputs={self.input_blob_: preproc_frame})
        infer_res = self.get_output()
        angle_y, angle_p, angle_r = self.preprocess_output(infer_res)

        return angle_y, angle_p, angle_r

    def get_output(self):
        """ Extract and return the output results as a dictionary """
        output_d = {}
        for o in self.output_blob_:
            infer_output = self.exec_network_.requests[0].outputs[o]
            output_d[o] = infer_output[0][0]
            logger.debug('  - extracting DNN output from blob (%s)' % o)
            logger.debug('  - output shape: {}'.format(infer_output.shape))

        return output_d

    def preprocess_input(self, image):
        '''
        preprocess before feeding the data into the model for inference
        # shape: [1x3x60x60] - An input image in the format [BxCxHxW]
        '''
        [n, c, h, w] = self.get_input_shape()

        converted_frame = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)

        # convert HWC into CHW
        converted_frame = converted_frame.transpose((2, 0, 1))
        # convert shape NCHW
        converted_frame = converted_frame.reshape((n, c, h, w))

        logger.debug(f'input [{image.shape}] -> converted_input [{converted_frame.shape}]')

        return converted_frame

    def preprocess_output(self, outputs):
        '''
        preprocess before feeding the output of this model to the next model
        # name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees)
        # name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
        # name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).
        '''
        angle_y = outputs['angle_y_fc']
        angle_p = outputs['angle_p_fc']
        angle_r = outputs['angle_r_fc']

        logger.debug(f'angle_y: {angle_y}')
        logger.debug(f'angle_p: {angle_p}')
        logger.debug(f'angle_r: {angle_r}')

        return angle_y, angle_p, angle_r

