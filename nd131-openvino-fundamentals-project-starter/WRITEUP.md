# Project Write-Up

reporter: jaeyong.rho@woven-planet.global

submission data: 2021/01/23

## Custom Layers

The process behind converting custom layers involves...

Some of the potential reasons for handling custom layers are...

## Steps for Running Application

Before we start the project, we should install Intel® OpenVINO™ toolkit on your environment first. If you do not have it, you need to install the toolkit on your PC (refer to the following page).

https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html

### Step 1: Select DNN Models for Application
First, we need to select the Deep Learning Neural Network (DNN) model to be executed in the target application. In this project, our purpose is to deploy an application at the edge to detect people appeared in the monitor and count the total number of detected people. I chose the below three FP32 models for the inference. This is because i) and ii) are mainly target for detecting a person (pedestrian), and iii) also support a pedestrian class and it is a famous object detection model with the high accuracy.

- i) person-detection-retail-0013 (FP32):
https://github.com/openvinotoolkit/open_model_zoo/blob/7d235755e2d17f6186b11243a169966e4f05385a/models/intel/person-detection-retail-0013/description/person-detection-retail-0013.md
- ii) pedestrian-detection-adas-0002 (FP32):
https://github.com/openvinotoolkit/open_model_zoo/blob/7d235755e2d17f6186b11243a169966e4f05385a/models/intel/pedestrian-detection-adas-0002/description/pedestrian-detection-adas-0002.md
- iii) ssd_mobilenet_v2_coco (FP32):
https://github.com/openvinotoolkit/open_model_zoo/blob/7d235755e2d17f6186b11243a169966e4f05385a/models/public/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco.md


### Step 2: Download Pre-Trained Models
Open Model Zoo provides a variety of Intel and public pre-trained models such as object detection, semantic segmentation, and so on. These network models can be downloaded by using Model Downloader (`downloader.py`) installed on Intel® OpenVINO™ toolkit.

```
python3 /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name [a keyword of model that you want to download] -o [save directory path]
```

Without an option for `--precisions`, all the available precisions models (FP32, FP16, and INT8) will be downloaded from the server. For instance, I downloaded a `person-detection-0201` model without specifiying the `--precisions` option, then `person-detection-0201.bin` (weight file) and `person-detection-0201.xml` (network definition file) for FP32, FP16, INT8 are saved in the output directory.

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
To implement the application, we need to check an input and output shape of network model. The input and output shapes can be refered in OpenVINO™ documentation. 

For `person-detection-0201`
- input: [1x3x320x544] ([BxCxHxW])
- output: [1, 1, N, 7] (where N is the number of detected bounding boxes. Each detection has the format [image_id, label, conf, x_min, y_min, x_max, y_max])

For `person-detection-0201`
- input: [1x3x384x672] ([BxCxHxW])
- output: [1, 1, N, 7] (where N is the number of detected bounding boxes. Each detection has the format [image_id, label, conf, x_min, y_min, x_max, y_max])

For `ssd_mobilenet_v2_coco`
- input: [1,300,300,3] ([BxHxWxC])
- output: [1, 1, N, 7] (where N is the number of detected bounding boxes. Each detection has the format [image_id, label, conf, x_min, y_min, x_max, y_max])

### Step 5: Implementation the Application
Please refer to below GitHub repository how we implemented the application. I will omit the detailed explanation for the codes here.

https://github.com/jaya-rho/udacity_project_1/tree/main/nd131-openvino-fundamentals-project-starter

### Step 6: Running the Application
#### 1) Connect to mosca server:
```
sudo node ./server.js
```
If 「Mosca server started」 is shown, it is connected correctly.

#### 2) Connect to GUI
```
cd webservice/ui
npm run dev
```
If 「webpack: Compiled successfully」is shown, it is connected correctly.

#### 3) Connect to ffserver
```
cd nd131-openvino-fundamentals-project-starte
sudo ffserver -f ./ffmpeg/server.conf
```
If such a「Tue Jan 19 22:43:00 2021 FFserver started」is shown, it is connected correctly.

#### 4) Run the application
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

## Comparison for Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Each of these use cases would be useful because...

- Entry Restriction into Convenience Store
to keep a social distance withing COVID-19, the number of people who can enter the convenience store is limited up to 10 at once, also a person can stary in the store up to 5 minutes. If a person who does not follow this rule, an alert will be announced.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
