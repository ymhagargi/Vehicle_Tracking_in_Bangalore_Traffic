import torch
import cv2
import numpy as np
from sort.sort import *
from deep_sort_realtime.deepsort_tracker import DeepSort

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

object_tracker = DeepSort(max_iou_distance=1, max_age=200,nms_max_overlap = 5)

def detect_cars(img, model, threshold = 0.5):
    detections = []
    pred = model(img)
    car_pred = pred.pred[0][pred.pred[0][:, 5] == 2]
    boxes = car_pred[:, :4].cpu().numpy()
    confidences = car_pred[:, 4].cpu().numpy()
    class_label = car_pred[:, 5].cpu().numpy()

    for i in range(len(class_label)):
        if(confidences[i] > threshold):
            cv2.rectangle(img, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])), (255,0, 0), 2)
            cv2.putText(img, "Car", (int(boxes[i][0])-10, int(boxes[i][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0, 0), 2)
            detections.append((boxes[i], confidences[i]))
    
    return detections

def update_track(tracks, count, track_ids):
    for t in tracks:
        if not t.is_confirmed():
            continue
        t_id = t.track_id
        if(t_id not in track_ids):
            count += 1
            track_ids.append(t_id)
        
        ltrb = t.to_ltrb()
        bbox = ltrb

    return track_ids, count

cap = cv2.VideoCapture('vid1.mp4')

track_ids = []
count = 0 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_cars(frame, model, 0.5)
    tracks = object_tracker.update_tracks(detections, frame=frame)
    track_ids, count = update_track(tracks, count, track_ids)
    cv2.putText(frame, f'Count: {int(count)}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)

    cv2.imshow('result', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()