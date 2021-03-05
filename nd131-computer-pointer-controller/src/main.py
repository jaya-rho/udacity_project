import cv2
from argparse import ArgumentParser

from logger import Logger
logger = Logger.get_logger('logger')

from face_detection import Model_FaceDetection
from facial_landmarks_detection import Model_FacialLandmarksDetection
from head_pose_estimation import Model_HeadPoseEstimation
from input_feeder import InputFeeder

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument('face_detec', type=str, help='model path for face_detection')
    parser.add_argument('facial_land', type=str, help='model path for facial_landmarks_model')
    parser.add_argument('head_pose', type=str, help='model path for head_pose_model')
    parser.add_argument('gaze_est', type=str, help='model path for gaze_estimation_model')
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
    parser.add_argument('--log_level', type=str, choices=['debug', 'info', 'warning', 'error', 'critical'])

    return parser

def main():

    # Grab command line args
    args = build_argparser().parse_args()

    # Create a logger object
    Logger('logger', args.log_level)

    # Check arguments
    logger.info(' Arguements:')
    logger.info(f'  - model #1: {args.face_detec}')
    logger.info(f'  - model #2: {args.facial_land}')
    logger.info(f'  - model #3: {args.head_pose}')
    logger.info(f'  - model #4: {args.gaze_est}')
    logger.info(f'  - input: {args.input}')
   
    # Load the input video
    input_feeder = InputFeeder("video", args.input)
    input_feeder.load_data()

    # Load Face Detection model
    net_fd = Model_FaceDetection(args.face_detec, args.device, args.cpu_extension) 
    net_fd.load_model()

    # Load Facial Landmarks Detection model
    net_fl = Model_FacialLandmarksDetection(args.facial_land, args.device, args.cpu_extension) 
    net_fl.load_model()
    
    # Load Facial Landmarks Detection model
    net_hp = Model_HeadPoseEstimation(args.head_pose, args.device, args.cpu_extension) 
    net_hp.load_model()

    # read the input data
    n_frame = 0
    for ret, frame in input_feeder.next_batch():

        if not ret:
            break

        logger.debug(f'frame #{n_frame:3d}: {frame.shape}')
        n_frame += 1

        #############################################
        # inference for face detection              #
        #############################################
        logger.info(f'** START the inference for face detection **')
        preproc_frame_fd = net_fd.preprocess_input(frame)
        net_fd.predict(preproc_frame_fd)

        # Wait for the result
        if net_fd.wait() == 0: # when the inference per frame finishes
            # Get the results of the inference request
            infer_result = net_fd.get_output()
            valid_bbox = net_fd.preprocess_output(infer_result, 0.3)
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

            # for debug, crop a face part
            cv2.imwrite(f"face_only_frame{n_frame}.jpg", cropped_face_frame)
            # for debug, draw a bbox on face
            cv2.rectangle(frame, (xmin_i, ymin_i), (xmax_i, ymax_i), (0, 0, 255), 3)
            cv2.imwrite(f"face_frame{n_frame}.jpg", frame)

        #############################################
        # inference for facial landmarks detection  #
        #############################################
        logger.info(f'** START the inference for facial landmarks detection **')
        preproc_frame_fl = net_fl.preprocess_input(cropped_face_frame)
        net_fl.predict(preproc_frame_fl)

        # Wait for the result
        if net_fl.wait() == 0: # when the inference per frame finishes
            # Get the results of the inference request
            infer_result_fl = net_fl.get_output()
            valid_eyes = net_fl.preprocess_output(infer_result_fl, 0.3)
            print(f'preproc shape: {preproc_frame_fl.shape}')
            preproc_frame_fl_h = preproc_frame_fl.shape[2]
            preproc_frame_fl_w = preproc_frame_fl.shape[3]

            left_eye_x_i = int(preproc_frame_fl_w * valid_eyes[0])
            left_eye_y_i = int(preproc_frame_fl_h * valid_eyes[1])
            right_eye_x_i = int(preproc_frame_fl_w * valid_eyes[2])
            right_eye_y_i = int(preproc_frame_fl_h * valid_eyes[3])

            logger.debug(f'[valid eye bbox on frame]')
            logger.debug(f'  left_eye x: {left_eye_x_i}')
            logger.debug(f'  left_eye y: {left_eye_y_i}')
            logger.debug(f'  right_eye x: {right_eye_x_i}')
            logger.debug(f'  right_eye y: {right_eye_y_i}')

            # NCHW -> HWC
            (n, c, h, w) = preproc_frame_fl.shape
            umat_preproc_frame_fl = preproc_frame_fl.reshape((c,h,w))
            umat_preproc_frame_fl = umat_preproc_frame_fl.transpose((1,2,0))

            cv2.circle(umat_preproc_frame_fl, (left_eye_x_i, left_eye_y_i), 2, (0,0,255), -1)
            cv2.circle(umat_preproc_frame_fl, (right_eye_x_i, right_eye_y_i), 2, (0,255,0), -1)
            cv2.imwrite(f'eye_frame{n_frame}.jpg', umat_preproc_frame_fl)

        #############################################
        # inference for head pose estimation        #
        #############################################
        logger.info(f'** START the inference for head pose estimation **')
        preproc_frame_hp = net_hp.preprocess_input(cropped_face_frame)
        net_hp.predict(preproc_frame_hp)

        # Wait for the result
        if net_hp.wait() == 0: # when the inference per frame finishes
            # Get the results of the inference request
            infer_result_hp = net_hp.get_output()
            valid_head_pose = net_hp.preprocess_output(infer_result_hp, 0.3)

if __name__ == '__main__':
    main()
