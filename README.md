# Object_Detection_using_OpenCV
This repository contains two Python scripts for performing object detection using OpenCV and a pre-trained model. One script is for image object detection, and the other is for video object detection. Both scripts use a MobileNet-based model pre-trained on the COCO dataset.

## Prerequisites

Before running the scripts, make sure you have the following prerequisites installed:

- Python 3.x
- OpenCV
- Matplotlib

You can install the required Python packages using pip:

```bash
pip install opencv-python matplotlib

## USAGE
   1.Image Object Detection
    -To perform object detection on an image, follow these steps:
       ~ Place the image you want to analyze in the same directory as the script.
       ~ Open the image.py script and specify the image file's name in the img variable:
         img = cv2.imread('path_to_image.jpg')

      ~Run the script:
       ```bash
       python image.py
       ~The annotated image with detected objects will be displayed.



    2.Video Object Detection
      -To perform object detection on a video stream or a video file, follow these steps:
          ~ Open the video.py script.
          ~ Specify the paths to the configuration file and frozen model in the config_file and frozen_model variables:
              config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
              frozen_model = 'frozen_inference_graph.pb'

          ~Specify the path to the labels file in the file_name variable:
              file_name = 'labels.txt'

          ~Run the script:
           ```bash
           python video.py

          ~The script will use your default camera or the specified video file for object detection, and the video stream with annotations will be displayed. Press 'q' 
           to exit the video stream.

## NOTES
~Use relative paths for file references to enhance code portability.
~Ensure that the frozen model file (frozen_inference_graph.pb) is available in the script folder.
~The provided code assumes 80 classes for object detection. Adjust the ClassInd <= 80 condition based on your specific class configuration.

  

