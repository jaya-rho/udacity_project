import cv2
from argparse import ArgumentParser

from face_detection import Model_FaceDetection
from facial_landmarks_detection import Model_FacialLandmarksDetection
from head_pose_estimation import Model_HeadPoseEstimation
from gaze_estimation import Model_GazeEstimation
from input_feeder import InputFeeder
from mouse_controller import MouseController
from visualizer import Visualizer

from logger import Logger
logger = Logger.get_logger('logger')


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

    # grab command line args
    args = build_argparser().parse_args()

    # create a logger object
    Logger('logger', args.log_level)

    # check arguments
    logger.info(' Arguements:')
    logger.info(f'  - model #1: {args.face_detec}')
    logger.info(f'  - model #2: {args.facial_land}')
    logger.info(f'  - model #3: {args.head_pose}')
    logger.info(f'  - model #4: {args.gaze_est}')
    logger.info(f'  - input: {args.input}')
   
    # load the input video
    input_feeder = InputFeeder("video", args.input)
    input_feeder.load_data()

    # mouse controller
    mc = MouseController('medium', 'fast')

    # load the models
    net_fd = Model_FaceDetection(args.face_detec, args.device, args.cpu_extension) 
    net_fl = Model_FacialLandmarksDetection(args.facial_land, args.device, args.cpu_extension) 
    net_hp = Model_HeadPoseEstimation(args.head_pose, args.device, args.cpu_extension) 
    net_ge = Model_GazeEstimation(args.gaze_est, args.device, args.cpu_extension) 
    # visualier
    vis = Visualizer()

    # read the input data
    frame_num = 0

    for ret, frame in input_feeder.next_batch():

        if not ret:
            break

        logger.debug(f'frame #{frame_num:3d}: {frame.shape}')
        frame_num += 1

        #############################################
        # step1: inference for face detection       #
        #############################################
        cropped_face_frame, face_coord = net_fd.predict(frame, frame_num)
        vis_frame = vis.draw_face_bbox(frame, face_coord, frame_num)

        ####################################################
        # step2: inference for facial landmarks detection  #
        ####################################################
        eyes_coord, cropped_left_eye, cropped_right_eye = net_fl.predict(cropped_face_frame)
        vis.draw_eye_bbox(cropped_face_frame, eyes_coord, frame_num)

        #############################################
        # step3: inference for head pose estimation #
        #############################################
        angle_y, angle_p, angle_r = net_hp.predict(cropped_face_frame)

        #############################################
        # step4: inference for gaze estimation      #
        #############################################
        hp_angle = [angle_y, angle_p, angle_r]
        mouse_coord, gaze_vector = net_ge.predict(cropped_left_eye, cropped_right_eye, hp_angle)
#        vis.draw_gaze(cropped_face_frame, gaze_vector, cropped_left_eye, cropped_right_eye, eyes_coord)

        # show an input image
        cv2.imshow('Visualize', vis_frame)

        # set speed
        if frame_num % 5 == 0:
            mc.move(mouse_coord[0], mouse_coord[1])

       
if __name__ == '__main__':
    main()

