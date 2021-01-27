# Project Write-Up

reporter: jaeyong.rho@woven-planet.global

submission date: 2021/01/23

## Custom Layers

Intel® OpenVINO™ toolkit supports many types of layers currently. However, there may be cases in which the supported layers do not match with the specific needs of a model. In this case, users can realize the specific functionality layer by implementing a custom layer.

According to `Custom Layers Guide` page, the Model Optimizer searches the list of known layers for each layer contained in the input model topology before building the model's internal representation, optimizing the model, and producing the Intermediate Representation files. The Inference Engine loads the layers from the input model IR files into the specified device plugin, which will search a list of known layer implementations for the device. If your topology contains layers that are not in the list of known layers for the device, the Inference Engine considers the layer to be unsupported and reports an error.

## Steps for Running Application

Before we start, we should install Intel® OpenVINO™ toolkit on our environment first (If you do not work on Udacity workspace). Please refer to below page to install the toolkit on your PC.

- https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html

### Step 1: Select DNN Models for Application
 First, we need to select the Deep Learning Neural Network (DNN) model to be executed for inference in the target application. In this project, our purpose is to deploy an application at the edge to detect people appeared in the monitor and count them. I chose the below five FP32 models for the inference from Open Model Zoo. 

- Model 1: `person-detection-retail-0013` (FP32):
https://github.com/openvinotoolkit/open_model_zoo/blob/7d235755e2d17f6186b11243a169966e4f05385a/models/intel/person-detection-retail-0013/description/person-detection-retail-0013.md
- Model 2: `pedestrian-detection-adas-0002` (FP32):
https://github.com/openvinotoolkit/open_model_zoo/blob/7d235755e2d17f6186b11243a169966e4f05385a/models/intel/pedestrian-detection-adas-0002/description/pedestrian-detection-adas-0002.md
- Model 3: `ssd_mobilenet_v2_coco` (FP32):
https://github.com/openvinotoolkit/open_model_zoo/blob/7d235755e2d17f6186b11243a169966e4f05385a/models/public/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco.md
- Model 4: `ssdlite_mobilenet_v2` (FP32):
https://github.com/openvinotoolkit/open_model_zoo/blob/7d235755e2d17f6186b11243a169966e4f05385a/models/public/ssdlite_mobilenet_v2/ssdlite_mobilenet_v2.md
- Model 5: `ssd_mobilenet_v1_coco` (FP32):
https://github.com/openvinotoolkit/open_model_zoo/blob/7d235755e2d17f6186b11243a169966e4f05385a/models/public/ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco.md

For pre-converted IR models, Models 1 and 2 are mainly target for detecting a person (pedestrian) and output the coordinates of the bounding box. For public models (which are not converted to IR), Model 3, 4, and 5 support a pedestrian class for detection and these are the famous object detection models with the high detection accuracy.

### Step 2: Download Pre-Trained Models
Open Model Zoo provides a variety of Intel and public pre-trained models such as object detection, semantic segmentation, and so on. These network models can be downloaded by using Model Downloader (`downloader.py`) installed on Intel® OpenVINO™ toolkit.

```
python3 /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name [a keyword of model that you want to download] -o [save directory path]
```

Without an option for `--precisions`, all the available precisions models (FP32, FP16, and INT8) will be downloaded from the server. For instance, I downloaded a `person-detection-retail-0013` model without specifiying the `--precisions` option, then `person-detection-retail-0013.bin` (weight file) and `person-detection-retail-0013.xml` (network definition file) for FP32, FP16, and INT8 are downloaded in the specified output directory.

### Step 3: Convert Tensorflow Model to Itermediate Representation
For a public pre-trained model such as ssd_mobilenet_v2_coco, we need to convert the downloaded Tensorflow model into a Itermediate Representation (IR). For instance, `ssd_mobilenet_v2_coco` models can be downloaded using the above `downloader.py` command. The downloaded package includes a Tensorflow model and config file, and so on. We need to convert this Tensorflow model to Intermediate Representation to enable to input the model into Intel® OpenVINO™ toolkit.

```
cd /opt/intel/openvino/deployment_tools/model_optimizer
python3 mo.py \
--input_model frozen_inference_graph.pb \
--tensorflow_object_detection_api_pipeline_config pipeline.config \
--reverse_input_channels \
--transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```
This command will convert the Tensorflow model and output the two files.
- frozen_inference_graph.xml
- frozen_inference_graph.bin

  (We changed the file name so that the model can be distinct from others)

### Step 4: Checking an Input/Output Shape of the Models
To implement the application including the image pre-process and DNN post-process, we need to know a shape and format of input and output of the network model. The input and output shapes and formats can be surveyed by OpenVINO™ documentation or using OpenVINO™ APIs.

For `person-detection-retail-0013`
- input: [1x3x320x544] ( [B x C x H x W] )
- output: [1, 1, N, 7] (where N is the number of detected bounding boxes. Each detection has the format [image_id, label, conf, x_min, y_min, x_max, y_max])

For `pedestrian-detection-adas-0002`
- input: [1x3x384x672] ( [B x C x H x W] )
- output: [1, 1, N, 7] (where N is the number of detected bounding boxes. Each detection has the format [image_id, label, conf, x_min, y_min, x_max, y_max])

For `ssd_mobilenet_v2_coco`
- input: [1,300,300,3] ( [B x H x W x C] )
- output: [1, 1, N, 7] (where N is the number of detected bounding boxes. Each detection has the format [image_id, label, conf, x_min, y_min, x_max, y_max])


For `ssd_mobilenet_v2_coco`
- input: [1,3,300,300] ( [B x C x H x W] )
- output: [1, 1, N, 7] (where N is the number of detected bounding boxes. Each detection has the format [image_id, label, conf, x_min, y_min, x_max, y_max])

For `ssd_mobilenet_v1_coco`
- input: [1,3,300,300] ( [B x C x H x W] )
- output: [1, 1, N, 7] (where N is the number of detected bounding boxes. Each detection has the format [image_id, label, conf, x_min, y_min, x_max, y_max])

### Step 5: Implementation the Application
Please refer to below the source codes how I implemented the application including the image pre-process and DNN post-process. I will omit the detailed explanation for the codes here.

- https://github.com/jaya-rho/udacity_project_1/tree/main/nd131-openvino-fundamentals-project-starter

### Step 6: Running the Application
Please install the required three components, refer to the below URL.
- https://github.com/udacity/nd131-openvino-fundamentals-project-starter
1) MQTT Mosca server
2) Node.js* Web server
3) FFmpeg server

#### (1) Connect to MQTT Mosca Server:
```
sudo node ./server.js
```
It is connected correctly if「Mosca server started」 is shown in a terminal.

#### 2) Connect to GUI Web Server
```
cd webservice/ui
npm run dev
```
It is connected correctly if「webpack: Compiled successfully」 is shown in a terminal.

#### 3) Connect to FFmpeg Server
```
cd nd131-openvino-fundamentals-project-starte
sudo ffserver -f ./ffmpeg/server.conf
```
It is connected correctly if such a 「Tue Jan 19 22:43:00 2021 FFserver started」 is shown in a terminal.

#### 4) Run the application (on CPU)
```
source /opt/intel/openvino_2021/bin/setupvars.sh
export CAMERA_FEED_SERVER="http://localhost:3004"
export MQTT_SERVER="ws://localhost:3002"

python3 main.py \
-i resources/Pedestrian_Detect_2_1_1.mp4 \
-m ../../network_models/person-detection-retail-0013/FP32/person-detection-retail-0013.xml \
-d CPU \
-pt 0.38 \
--log_level info | ffmpeg \
-v warning \
-f rawvideo \
-pixel_format bgr24 \
-video_size 768x432 \
-framerate 24 \
-i - http://0.0.0.0:3004/fac.ffm 
```
#### 5) View the Results on Browser
```
http://0.0.0.0:3000/
```
![A sample capture of the application](https://github.com/jaya-rho/udacity_project_1/blob/main/nd131-openvino-fundamentals-project-starter/images/intel_project1_on_mqtt.png)

## Comparison for Model Performance

We compared the performance of the model conversion (optimization) in terms of i) model size and ii) average inference time using a network model `ssd_mobilenet_v2_coco` (FP32).

- Model Size:
```
before conversion: 
   ssd_mobilenet_v2_coco/*: 202 MB

after conversion: 
   ssd_mobilenet_v2_coco.bin: 65 MB
   ssd_mobilenet_v2_coco.xml: 256 KB
```
Before the model conversion, the model size which includes all the files in the downloaded `ssd_mobilenet_v2_coco/` is almost 202 MB. However, the model size decreases to about 65.3 MB after adapting a model conversion (optimization) tool.

- Inference Time (an average for 1,000 iterations):
```
before conversion:
   the inference time in average: 22.41 ms
after conversion:  
   the inference time in average: 14.22 ms
```

We run the inference for `ssd_mobilenet_v2_coco` model with input data size [1,300,300,3] on CPU, and get the average inference time for 1,000 iterations. To measure the inference time with a Tensorflow model, we used a below python program.

- https://github.com/jaya-rho/udacity_project_1/blob/main/nd131-openvino-fundamentals-project-starter/infer_tf_model.py

The results are shown in above. Even though the converted model has smaller size compared to the original one (Tensorflow model), the inference time in average decreases from 22.41 ms to 14.22 ms in FP32 precision. 


## Assess Model Use Cases

Some of the potential use cases of the people counter app are described below.

1) Admission Restriction in Convenience Store

      It is very important to keep a social distance and to avoid a place where is very crowded by people due to COVID-19. Assume that a convenience store begins to limit the number of people who can enter the store up to 10 at the same time. Also, a time that a person can stay in the store after he or she enters is 15 minutes in the maximum. If a person who does not follow these rules, some cautions will be announced.

2) Survey for How Long It Takes to Solve the Mathematic Question?

      Assume that I am making a question for the mathematics exam. This question seems like pretty easy for me, but I do not know how difficult this question for high school students. So I invite a hundred of students to accept this survey, and a student who accepts it will take an exam in a classroom alone. He or she can exit the room if he or she finishes the exam. The people counter application can be useful in this situation because I can know how many student took this exam and how much time a student spend to get the answer in average.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model.

- Lighting and Focal Length of the camera can significantly reduces the accuracy of the model. The camera condition such as foggy lenses and an installed camera-angle are very important issue related to the model detection accuracy.

- The model used in the application should be well trained with the images which are similar with the real-environment scenes.

## Model Research

In investigating potential people counter models, I tried each of the following three models. We used the same input file named `resources/Pedestrian_Detect_2_1_1.mp4` for the inference of three models. To determine an appropriate confidence threshold for each model, we seached the maximum value as the confidence threshold where the number of detected object (person) per one frame does not exceed one. This is because the number of person per each frame in the input data (movie) is not over one (a person appears in a sequence manner).

- Model 1: `person-detection-retail-0013` (pre-converted IR model)
  - https://github.com/openvinotoolkit/open_model_zoo/blob/7d235755e2d17f6186b11243a169966e4f05385a/models/intel/person-detection-retail-0013/description/person-detection-retail-0013.md
  - I downloaded with the following command.
    ```
    python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name person-detection-retail-0013 -o .
    ```
  - We did not have to convert to an Intermediate Representation (IR) since it is already converted when I downloaded from the server.

- Model 2: `pedestrian-detection-adas-0002` (pre-converted IR model)
  - https://github.com/openvinotoolkit/open_model_zoo/blob/7d235755e2d17f6186b11243a169966e4f05385a/models/intel/pedestrian-detection-adas-0002/description/pedestrian-detection-adas-0002.md
  - I downloaded with the following command.
    ```
    python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name pedestrian-detection-adas-0002 -o .
    ```
  - We did not have to convert to an Intermediate Representation (IR) since it is already converted when I downloaded from the server.

- Model 3: `ssd_mobilenet_v2_coco` (NOT pre-converted model)
  - https://github.com/openvinotoolkit/open_model_zoo/blob/7d235755e2d17f6186b11243a169966e4f05385a/models/public/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco.md
  - I downloaded with the following command.
    ```
    python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name ssd_mobilenet_v2_coco -o .
    ```
  - I converted the model to an Intermediate Representation with the following command.
    ```
    cd /opt/intel/openvino/deployment_tools/model_optimizer
    python3 mo.py \
    --input_model ssd_mobilenet_v2_coco/frozen_inference_graph.pb \
    --tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v2_coco/pipeline.config \
    --reverse_input_channels \
    --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
    ```

- Model 4: `ssdlite_mobilenet_v2` (NOT pre-converted model)
  - https://github.com/openvinotoolkit/open_model_zoo/blob/7d235755e2d17f6186b11243a169966e4f05385a/models/public/ssdlite_mobilenet_v2/ssdlite_mobilenet_v2.md
  - I downloaded with the following command.
    ```
    python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name ssdlite_mobilenet_v2 -o .
    ```
  - I converted the model to an Intermediate Representation with the following command.
    ```
    cd /opt/intel/openvino/deployment_tools/model_optimizer
    python3 mo.py \
    --input_model ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb \
    --tensorflow_object_detection_api_pipeline_config ssdlite_mobilenet_v2_coco_2018_05_09/pipeline.config \
    --reverse_input_channels \
    --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
    ```

- Model 5: `ssd_mobilenet_v1_coco` (NOT pre-converted model)
  - https://github.com/openvinotoolkit/open_model_zoo/blob/7d235755e2d17f6186b11243a169966e4f05385a/models/public/ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco.md
  - I downloaded with the following command.
    ```
    python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name ssd_mobilenet_v1_coco -o .
    ```
  - I converted the model to an Intermediate Representation with the following command.
    ```
    cd /opt/intel/openvino/deployment_tools/model_optimizer
    python3 mo.py \
    --input_model ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb \
    --tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v1_coco_2018_01_28/pipeline.config \
    --reverse_input_channels \
    --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
    ```

- Summary for Performance Comparison
  - | Model Name | Precision | Model Size [MB] | Average Inference Time per Frame [ms] (※)| Total Count of People| Confidence Threshold|
    | --- | --: | ---: | ---: | ---: | ---: | ---: | 
    | person-detection-retail-0013 | FP32 | 3.2 MB | 10.99 ms | 6 | 0.9 |
    | pedestrian-detection-adas-0002 | FP32 | 4.7 MB | 19.97 ms | 6 | 0.9 |
    | ssd_mobilenet_v2_coco | FP32 | 65 MB | 21.72 ms | 13 | 0.4 |
    | ssdlite_mobilenet_v2 | FP32 | 18 MB | 13.35 ms | 11 | 0.5 |
    | ssd_mobilenet_v1_coco | FP32 | 27 MB | 16.15 ms | 18 | 0.3 |
    |||||||

    (※ Inference time: 
a time from the start time of infer_network.exec_net() to the finish time of infer_network.wait() == 0; thus, this time does not include the image pre-process and post-process of DNN)

     - For all the models, we searched the optimized confidence threshold value where the detection performance can improve through [0.0, 1.0]. We found a threshold value where mis-detection (e.g. a desk is recognized as a person class) has not occurred and can keep detecting a person correctly. Please refer to `Confidence Threshold` in the table.

     - Conclusion: I select **Model 1** ( **`person-detection-retail-0013`** ) to use in the target people count application with consideration of the three below things.

       - i) Low memory consumption:
          - As shown in the table, person-detection-retail-0013 requires only 3.2 MB to deploy the model on the edge computer. In contrast, ssd_mobilenet_v2_coco needs 65 MB and ssd_mobilenet_v1_coco requires 27 MB, and so on. We can say that person-detection-retail-0013 model is very small memory requirement compared to other models.
       - ii) High detection accuracy (a total count of people):
          - person-detection-retail-0013 and pedestrian-detection-adas-0002 can detect a person (class) with a higher confidence compared to other models. The results of the total count of people for both models were six which is a correct answer.
       - iii) Fast inference time:
          - A model which has the smallest inference time per one frame is person-detection-retail-0013. It can inference one image within 10.99ms in average, and this value is the fastest value among all the other models.

