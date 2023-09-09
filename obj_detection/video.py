import cv2

# Description:
# This script demonstrates object detection using a pre-trained model on a video stream.
# Note: Whenever possible, use relative paths for better code portability.

# Define the paths to the configuration file and frozen model. Replace with your actual file paths.
config_file = 'path_to_config_file.pbtxt'
# Use 'frozen_inference_graph.pb' from the given folder.
frozen_model = 'frozen_inference_graph.pb'

# Load the pre-trained model and labels
model = cv2.dnn.DetectionModel(frozen_model, config_file)

# Load class labels from a text file. Replace with your labels file path.
classLabels = []
file_name = 'labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

# Set input parameters for the model
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Open the video file or use the default camera (camera index 0). Replace with your video file path.
cap = cv2.VideoCapture('path_to_video.mp4')

if not cap.isOpened():
    cap = cv2.VideoCapture(0)  # Use the default camera (camera index 0) if the video file cannot be opened

if not cap.isOpened():
    raise IOError("Cannot open video")

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

    if len(ClassIndex):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= len(classLabels):  # Assuming you have a specific number of classes
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)

        cv2.imshow('Object Detection Tutorial', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
