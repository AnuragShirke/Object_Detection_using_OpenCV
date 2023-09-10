import cv2
import matplotlib.pyplot as plt

# Description:
# This script demonstrates object detection using a pre-trained model on an image.
# Note: Whenever possible, use relative paths for better code portability.

# Define the paths to the configuration file and frozen model. Replace with your actual file paths.
config_file = 'path_to_config_file.pbtxt'
frozen_model = 'path_to_frozen_model.pb'

# Load the pre-trained model and labels
model = cv2.dnn.DetectionModel(frozen_model, config_file)

# Load class labels from a text file. Replace with your labels file path.
classLabels = []
file_name = 'path_to_labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

# Set input parameters for the model
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Load and process the image. Replace with your image file path.
img = cv2.imread('path_to_image.jpg')

# Perform object detection on the image
ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

# Define font settings for labeling detected objects
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

# Draw bounding boxes and labels on the image
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)

# Display the annotated image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

