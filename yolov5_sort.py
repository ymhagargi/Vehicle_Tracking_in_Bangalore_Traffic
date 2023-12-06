import torch
import cv2
import numpy as np
from sort.sort import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True).to(device)

model.eval()

cap = cv2.VideoCapture('vid1.mp4')
object_tracker = Sort(max_age=2500, min_hits=2, iou_threshold=0.2)
bounding_boxes_ids = np.array([])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    pred = model(frame)
    car_pred = pred.pred[0][pred.pred[0][:, 5] == 2]
    boxes = car_pred[:, :4].cpu().numpy()
    confidences = car_pred[:, 4].cpu().numpy()
    class_label = car_pred[:, 5].cpu().numpy()

    bounding_boxes = []
    for i in range(len(class_label)):
        if(confidences[i] > 0.5):
            bounding_boxes.append(boxes[i])

    if(len(bounding_boxes) != 0):
        bounding_boxes = np.array(bounding_boxes)
        tracks = object_tracker.update(bounding_boxes)

        for i in range(tracks.shape[0]):
            bounding_boxes_ids = np.append(bounding_boxes_ids, np.array([tracks[i][4]]))
            
        count = np.unique(bounding_boxes_ids).shape[0]
        temp = count

        for i in range(bounding_boxes.shape[0]):
            cv2.rectangle(frame, (int(bounding_boxes[i][0]), int(bounding_boxes[i][1])), (int(bounding_boxes[i][2]), int(bounding_boxes[i][3])), (255,0, 0), 2)
            cv2.putText(frame, "Car", (int(bounding_boxes[i][0])-10, int(bounding_boxes[i][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0, 0), 2)

            cv2.putText(frame, f'Count: {int(count)}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)    

            cv2.imshow('result', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()