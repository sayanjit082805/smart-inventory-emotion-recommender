# detect_objects.py
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(frame):
    results = model(frame)
    labels = results.names
    detections = results.pred[0]
    detected = []

    for *box, conf, cls in detections:
        label = labels[int(cls)]
        detected.append(label)

    return list(set(detected))  # remove duplicates
