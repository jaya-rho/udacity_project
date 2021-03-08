'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2

from openvino.inference_engine import IENetwork, IECore
from base_model import Model

from logger import Logger
logger = Logger.get_logger('logger')

class Model_FaceDetection(Model):
    '''
    Class for the Face Detection Model.
    https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_0001_description_face_detection_adas_0001.html
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        class initialization
        '''
        super().__init__(model_name, device, extensions) # initialize the base class

    def predict(self, frame, n_frame, prob_thres=0.3):
        """This method is meant for running predictions on the input image"""
        
        preproc_frame = self.preprocess_input(frame)
#        self.exec_network_.start_async(request_id=request_id,inputs={self.input_blob_: image})
        self.exec_network_.infer(inputs={self.input_blob_: preproc_frame})
        infer_res = self.get_output()
        cropped_face_frame, face_coord = self.preprocess_output(infer_res, frame, n_frame, prob_thres)

        return cropped_face_frame, face_coord

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

    def preprocess_output(self, outputs, frame, n_frame, conf_thres):
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

        if len(detected_bboxes) == 0:
            # if no any face is found
            return None, None

        top_detected_bbox = detected_bboxes[0]

        # frame shape: (height, width, channel)
        frame_h = frame.shape[0]
        frame_w = frame.shape[1]

        xmin_i = int(frame_w * valid_bbox[0])
        xmax_i = int(frame_w * valid_bbox[2])
        ymin_i = int(frame_h * valid_bbox[1])
        ymax_i = int(frame_h * valid_bbox[3])

        logger.debug(f'[valid face bbox on frame]')
        logger.debug(f'  xmin: {xmin_i}')
        logger.debug(f'  xmax: {xmax_i}')
        logger.debug(f'  ymin: {ymin_i}')
        logger.debug(f'  ymax: {ymax_i}')

        cropped_face_frame = frame[ymin_i:ymax_i, xmin_i:xmax_i]

        return cropped_face_frame, [xmin_i, ymin_i, xmax_i, ymax_i]

