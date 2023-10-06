import cv2
import numpy as np
import os
import requests

weights_url = "https://pjreddie.com/media/files/yolov3.weights"
config_url = "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true"
names_url = "https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true"

weights_file = "yolov3.weights"
config_file = "yolov3.cfg"
names_file = "coco.names"

def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download {filename}")

if not os.path.exists(weights_file):
    download_file(weights_url, weights_file)

if not os.path.exists(config_file):
    download_file(config_url, config_file)

if not os.path.exists(names_file):
    download_file(names_url, names_file)

net = cv2.dnn.readNet(weights_file, config_file)

with open(names_file, "r") as f:
    classes = f.read().strip().split('\n')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x, center_y, width, height = map(int, obj[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]))
                x, y = center_x - width // 2, center_y - height // 2
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
