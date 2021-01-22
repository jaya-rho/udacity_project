source /opt/intel/openvino_2021/bin/setupvars.sh
export CAMERA_FEED_SERVER="http://localhost:3004"
export MQTT_SERVER="ws://localhost:3002"
#python3 main.py -m ../../network_models/person-detection-retail-0013/FP32/person-detection-retail-0013.xml -i resources/Pedestrian_Detect_2_1_1.mp4
python3 main.py \
-i resources/Pedestrian_Detect_2_1_1.mp4 \
-m ../../network_models/person-detection-retail-0013/FP32/person-detection-retail-0013.xml \
-d CPU \
-pt 0.38 | ffmpeg \
-v warning \
-f rawvideo \
-pixel_format bgr24 \
-video_size 768x432 \
-framerate 24 \
-i - http://0.0.0.0:3004/fac.ffm
