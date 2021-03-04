from argparse import ArgumentParser

from logger import Logger
logger = Logger.get_logger('logger')

from face_detection import Model_FaceDetection

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

    logger.info(' Arguements:')
    logger.info(f'  - model #1: {args.face_detec}')
    logger.info(f'  - model #2: {args.facial_land}')
    logger.info(f'  - model #3: {args.head_pose}')
    logger.info(f'  - model #4: {args.gaze_est}')
    logger.info(f'  - input: {args.input}')
   
    # self, model_name, device='CPU', extensions=None
    m_fd = Model_FaceDetection(args.face_detec, args.device, args.cpu_extension) 

if __name__ == '__main__':
    main()
