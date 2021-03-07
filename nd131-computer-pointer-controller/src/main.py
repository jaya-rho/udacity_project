import cv2
import time

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
    load_fd_start = time.time()
    net_fd = Model_FaceDetection(args.face_detec, args.device, args.cpu_extension) 
    load_fd_end = time.time()
    load_fl_start = time.time()
    net_fl = Model_FacialLandmarksDetection(args.facial_land, args.device, args.cpu_extension) 
    load_fl_end = time.time()
    load_hp_start = time.time()
    net_hp = Model_HeadPoseEstimation(args.head_pose, args.device, args.cpu_extension) 
    load_hp_end = time.time()
    load_ge_start = time.time()
    net_ge = Model_GazeEstimation(args.gaze_est, args.device, args.cpu_extension) 
    load_ge_end = time.time()

    logger.info(f'model load time [ms]')
    logger.info(f'   face detection model       : {load_fd_end - load_fd_start:6.3f} [ms]')
    logger.info(f'   facial landmarks model     : {load_fl_end - load_fl_start:6.3f} [ms]')
    logger.info(f'   head pose estimation model : {load_hp_end - load_hp_start:6.3f} [ms]')
    logger.info(f'   gaze estimation model      : {load_ge_end - load_ge_start:6.3f} [ms]')
    logger.info(f'   TOTAL                      : {load_ge_end - load_fd_start:6.3f} [ms]')

    # visualier
    vis = Visualizer()

    # read the input data
    frame_num = 0

    infer_times_d = {'fd':[], 'fl':[], 'hp':[], 'ge':[]}

    for ret, frame in input_feeder.next_batch():

        if not ret:
            break

        key_pressed = cv2.waitKey(60)

        logger.debug(f'frame #{frame_num:3d}: {frame.shape}')
        frame_num += 1

        # inference for face detection
        infer_fd_start = time.time()
        cropped_face_frame, face_coord = net_fd.predict(frame, frame_num)
        infer_times_d['fd'].append(time.time() - infer_fd_start)

        # if no face has been detected, skip the frame
        if cropped_face_frame is None or face_coord is None:
            logger.warning(f'No face has been founded on frame #{frame_num}. Neither of cropped_face_frame nor face_coord is None')
            continue

        vis.draw_face_bbox(frame, face_coord, frame_num)

        # inference for facial landmarks detection
        infer_fl_start = time.time()
        eyes_coord, cropped_left_eye, cropped_right_eye = net_fl.predict(cropped_face_frame)
        infer_times_d['fl'].append(time.time() - infer_fl_start)
        vis_eye_gaze = vis.draw_eye_bbox(cropped_face_frame, eyes_coord, frame_num)

        # inference for head pose estimation
        infer_hp_start = time.time()
        angle_y, angle_p, angle_r = net_hp.predict(cropped_face_frame)
        infer_times_d['hp'].append(time.time() - infer_hp_start)
        hp_angle = [angle_y, angle_p, angle_r]

        # inference for gaze estimation
        infer_ge_start = time.time()
        mouse_coord, gaze_vector = net_ge.predict(cropped_left_eye, cropped_right_eye, hp_angle)
        infer_times_d['ge'].append(time.time() - infer_ge_start)
        vis.draw_gaze(cropped_face_frame, gaze_vector, cropped_left_eye, cropped_right_eye, eyes_coord, frame_num)

        if key_pressed == 27:
            break

        # show the results
        cv2.startWindowThread()
        cv2.namedWindow("Visualize")
        cv2.imshow('Visualize', frame)

        # control the mouse
        if (frame_num % 5) == 0:
            mc.move(mouse_coord[0], mouse_coord[1])
            logger.debug(f'Move mouse cursor to [x:{mouse_coord[0]}, y:{mouse_coord[1]}]')

    logger.info(f'model inference time [ms]')
    logger.info(f'   face detection model')
    logger.info(f'         average : {sum(infer_times_d["fd"])/len(infer_times_d["fd"])*1000:6.3f} [ms]')
    logger.info(f'         maximum : {max(infer_times_d["fd"])*1000:6.3f} [ms]')
    logger.info(f'   facial landmarks model')
    logger.info(f'         average : {sum(infer_times_d["fl"])/len(infer_times_d["fl"])*1000:6.3f} [ms]')
    logger.info(f'         maximum : {max(infer_times_d["fl"])*1000:6.3f} [ms]')
    logger.info(f'   head pose estimation model')
    logger.info(f'         average : {sum(infer_times_d["hp"])/len(infer_times_d["hp"])*1000:6.3f} [ms]')
    logger.info(f'         maximum : {max(infer_times_d["hp"])*1000:6.3f} [ms]')
    logger.info(f'   gaze estimation model')
    logger.info(f'         average : {sum(infer_times_d["ge"])/len(infer_times_d["ge"])*1000:6.3f} [ms]')
    logger.info(f'         maximum : {max(infer_times_d["ge"])*1000:6.3f} [ms]')

    # release the resources
    input_feeder.close()
    cv2.destroyAllWindows
       
if __name__ == '__main__':
    main()

