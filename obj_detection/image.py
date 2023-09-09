import cv2

# Description:
# This script demonstrates object detection using a pre-trained model on a video stream.
# Note: Use relative paths for better code portability. Ensure the frozen model is available in the script folder.

# Define the paths to the configuration file and frozen model (use 'frozen_inference_graph.pb' from the folder).
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

# Load the pre-trained model and labels
model = cv2.dnn.DetectionModel(frozen_model, config_file)

# Load class labels from a text file (replace with your labels file).
classLabels = []
file_name = 'labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

# Set input parameters for the model
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Open the video file or use the default camera (camera index 0)
cap = cv2.VideoCapture(0)  # Use the default camera (camera index 0)

if not cap.isOpened():
    raise IOError("Cannot open video")

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

    if len(ClassIndex):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:  # Assuming you have 80 classes
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)

        cv2.imshow('Object Detection Tutorial', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
