source /opt/intel/openvino_2021/bin/setupvars.sh
export CAMERA_FEED_SERVER="http://localhost:3004"
export MQTT_SERVER="ws://localhost:3002"
#XML_MODEL="../../network_models/person-detection-retail-0013/FP32/person-detection-retail-0013.xml"
#XML_MODEL="../../network_models/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco.xml"
XML_MODEL="../../network_models/pedestrian-detection-adas-0002/pedestrian-detection-adas-0002.xml"
INPUT_MV="resources/Pedestrian_Detect_2_1_1.mp4"

python3 main.py \
-i $INPUT_MV \
-m $XML_MODEL \
-d CPU \
--log_level info \
-pt 0.9 | ffmpeg \
-v warning \
-f rawvideo \
-pixel_format bgr24 \
-video_size 768x432 \
-framerate 24 \
-i - http://0.0.0.0:3004/fac.ffm
