#usage: main.py [-h] -i {video,image,camera} -ip INPUT_PATH [-l CPU_EXTENSION]
#               [-d DEVICE] [--visualize]
#               [--log_level {debug,info,warning,error,critical}]
#               face_detec facial_land head_pose gaze_est
#
#positional arguments:
#  face_detec            model path for face_detection
#  facial_land           model path for facial_landmarks_model
#  head_pose             model path for head_pose_model
#  gaze_est              model path for gaze_estimation_model
#
#optional arguments:
#  -h, --help            show this help message and exit
#  -i {video,image,camera}, --input_type {video,image,camera}
#                        Specify the input data type
#  -ip INPUT_PATH, --input_path INPUT_PATH
#                        Specifiy the input data path
#  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
#                        MKLDNN (CPU)-targeted custom layers.Absolute path to a
#                        shared library with thekernels impl.
#  -d DEVICE, --device DEVICE
#                        Specify the target device to infer on: CPU, GPU, FPGA
#                        or MYRIAD is acceptable. Sample will look for a
#                        suitable plugin for device specified (CPU by default)
#  --visualize
#  --log_level {debug,info,warning,error,critical}
#MODEL_FD='../../../models/intel/face-detection-adas-0001/FP16/face-detection-adas-0001'
MODEL_FD='../../../models/intel/face-detection-adas-0001/FP16-INT8/face-detection-adas-0001'
#MODEL_FD='../../../models/intel/face-detection-adas-0001/FP32/face-detection-adas-0001'

#MODEL_FL='../../../models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009'
MODEL_FL='../../../models/intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009'
#MODEL_FL='../../../models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009'

#MODEL_HP='../../../models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001'
MODEL_HP='../../../models/intel/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001'
#MODEL_HP='../../../models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001'

#MODEL_GE='../../../models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002'
MODEL_GE='../../../models/intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002'
#MODEL_GE='../../../models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002'

source /opt/intel/openvino_2021/bin/setupvars.sh
#python3 main.py $MODEL_FD $MODEL_FL $MODEL_HP $MODEL_GE -i ../bin/demo.mp4 --log_level debug
python3 main.py $MODEL_FD $MODEL_FL $MODEL_HP $MODEL_GE -i video -ip ../bin/demo.mp4 --log_level info --visualize
