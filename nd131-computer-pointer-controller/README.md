# Computer Pointer Controller

In this project, we build an application to control a mouse pointer by the gaze of user's eyes. As shown in the below flow chart, first, we need to detect a face from the input image. With the cropped face iamge, the left and right eyes can be detected by a landmark detection model, and the head pose angles can be estimated from a head pose estimation model. With the results from the previous two models, a gaze estimation model calculates the direction of mouse pointer movement.

![entire_flow](https://video.udacity-data.com/topher/2020/April/5e923081_pipeline/pipeline.png)

## Project Set Up and Installation
### i) Download and Intall OpenVino Toolkit
Intel® OpenVINO™ toolkit must be installed on the local development PC previously. I installed with the following instruction.
https://docs.openvinotoolkit.org/latest/index.html

### ii) Download the Models
The below four models are required to implement the mouse pointer controller application.

1. face detection model:
    - reference page:
 https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_0001_description_face_detection_adas_0001.html
    - download command:
        ```
        $ python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-0001" -o .
        ```

2. facial landmarks detection model:
    - reference page: 
    https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html
    - download command:
        ```
        $ python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009" -o .
        ```

3. head pose estimation model:
    - reference page: 
    https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html
    - download command:
        ```
        $ python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001" -o .
        ```

4. gaze estimation model:
    - reference page:
    https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html
    - download command:
        ```
        $ python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002" -o .
        ```

## Demo
1. Install the required python packages
    ```
    pip3 install -r requirements.txt
    ```

2. Run the application
    ```
    cd nd131-computer-pointer-controller/src
    sh run.sh
    ```

    We need to modify the paths for model and input movie file according to the local environment in run.sh. 

    ```
    MODEL_FD='../../../models/intel/face-detection-adas-0001/FP16/face-detection-adas-0001'
    MODEL_FL='../../../models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009'
    MODEL_HP='../../../models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001'
    MODEL_GE='../../../models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002'

    source /opt/intel/openvino_2021/bin/setupvars.sh
    python3 main.py $MODEL_FD $MODEL_FL $MODEL_HP $MODEL_GE -i ../bin/demo.mp4 --log_level info
    ```

3. Check the result

## Documentation

Following are commands line arguments that can use for while running the main.py file.
```
usage: main.py [-h] -i INPUT [-l CPU_EXTENSION] [-d DEVICE]
               [--log_level {debug,info,warning,error,critical}]
               face_detec facial_land head_pose gaze_est

positional arguments:
  face_detec            model path for face_detection
  facial_land           model path for facial_landmarks_model
  head_pose             model path for head_pose_model
  gaze_est              model path for gaze_estimation_model

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to image or video file
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a
                        shared library with thekernels impl.
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on: CPU, GPU, FPGA
                        or MYRIAD is acceptable. Sample will look for a
                        suitable plugin for device specified (CPU by default)
  --log_level {debug,info,warning,error,critical}
```

## Benchmarks

1. Performance Environment

    We used the demo movie file which is included in \bin folder as a default, and we used that file for testing the application and measure the performance. The local PC where we used for the performance test has the below specification. 
    - Intel Core™ i5-6200U CPU @ 2.30GHz × 4
    - 16 GB 3733 MHz LPDDR4X

2. Model Size

    | Model Name | Precision | Model Size [MB] |
    | --- | --: | ---: | ---: | ---: | ---: | 
    | face-detection-adas-0001 | FP16 | 2.3 MB |
    |  | FP16-INT8 | 1.6 MB |
    |  | FP32 | 4.3 MB |
    | landmarks-regression-retail-0009 | FP16 | 0.4 MB |
    |  | FP16-INT8 | 0.3 MB |
    |  | FP32 | 0.8 MB |
    | head-pose-estimation-adas-0001 | FP16 | 3.7 MB |
    |  | FP16-INT8 | 2.0 MB |
    |  | FP32 | 7.3 MB |
    | gaze-estimation-adas-0002 | FP16 | 3.6 MB |
    |  | FP16-INT8 | 2.0 MB |
    |  | FP32 | 7.2 MB |
    |||||

3. Inference Time

    | Model Name | Precision | Average Inference Time [ms] | Maximum Inference Time [ms] |
    | --- | --: | ---: | ---: | ---: | ---: | 
    | face-detection-adas-0001 | FP16 | 21.569 ms | 49.784 ms |
    |  | FP16-INT8 | 21.999 ms | 49.585 ms |
    |  | FP32 | 21.573 ms | 49.685 ms |
    | landmarks-regression-retail-0009 | FP16 | 0.846 ms | 1.133 ms |
    |  | FP16-INT8 | 0.817 ms | 1.036 ms |
    |  | FP32 | 0.858 ms | 1.207 ms |
    | head-pose-estimation-adas-0001 | FP16 | 1.351 ms | 2.099 ms |
    |  | FP16-INT8 |  1.228 ms | 3.468 ms |
    |  | FP32 | 1.439 ms | 2.302 ms |
    | gaze-estimation-adas-0002 | FP16 | 0.999 ms | 1.219 ms |
    |  | FP16-INT8 | 0.782 ms | 1.034 ms |
    |  | FP32 | 1.012 ms | 1.262 ms |
    |||||

3. Model Load Time

    | Model Name | Precision | Model Load Time [ms] |
    | --- | --: | ---: | ---: | ---: | ---: | 
    | face-detection-adas-0001 | FP16 | 0.209 ms |
    |  | FP16-INT8 | 0.258 ms |
    |  | FP32| 0.205 ms |
    | landmarks-regression-retail-0009 | FP16 | 0.051 ms |
    |  | FP16-INT8 | 0.093 ms |
    |  | FP32 | 0.051 ms |
    | head-pose-estimation-adas-0001 | FP16 | 0.073 ms |
    |  | FP16-INT8 | 0.106 ms |
    |  | FP32 | 0.118 ms |
    | gaze-estimation-adas-0002 | FP16 | 0.074 ms |
    |  | FP16-INT8 | 0.154 ms |
    |  | FP32 | 0.115 ms |
    |||||

## Results
For this project, we can use models in three precisions: FP32, FP16, and FP32-INT8. We compared the performance in three terms, i) model size, ii) inference time, and iii) load time for each precision of the four models.

- For model size, a model whose precision is FP16-INT8 has smaller size than the size of FP16 and FP32-precision models.

- The inference time on CPU for the three FP16, FP16-INT8, and FP32 precision models are not different that much in maximum and average. 

- For face detection and facial landmarks detection models, FP32 precision is slightly faster than other precisions. The load times for FP16 precision head pose estimation and gaze estimation models are better than FP16-INT8 and FP32 models.

## Stand Out Suggestions
- I compared the performance of the three different precision models in the three terms such as i) memory size, ii) inference time, and iii) load time to discover which is the best model precision model to use in this application

### Async Inference
- We did not implement the inference API with an asynchronization mode.

### Edge Cases
- Our application suppose that the exactly one face should appear in each frame. If there are multiple people (faces) in an input image, we need to modify and extend the post-processing of each module.
- Our application supposes that there is at least more than one face detected. If there is no any detected face from the input image, the frame should be skip and the remaining model inference should be skipped.
