MODEL_FD='../../models/intel/face-detection-adas-0001/FP16/face-detection-adas-0001'
MODEL_FL='../../models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009'
MODEL_HP='../../models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001'
MODEL_GE='../../models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002'

source /opt/intel/openvino_2021/bin/setupvars.sh
python3 main.py $MODEL_FD $MODEL_FL $MODEL_HP $MODEL_GE -i ../bin/demo.mp4 --log_level debug
