import torch
import torchvision
import cv2
import numpy as np
from sort.sort import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)

model.eval()

cap = cv2.VideoCapture('vid1.mp4')

object_tracker = Sort(max_age=300, min_hits=2)
bounding_boxes_ids = np.array([])

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

while cap.isOpened():
    
    ret, frame = cap.read()

    if not ret:
        break
    

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    frame1 = transform(frame).to(device)

    with torch.no_grad():
        pred = model([frame1])
    
    pred_class1 = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())] # Get the Prediction Score
    pred_boxes1 = [[i[0], i[1], i[2], i[3]] for i in list(pred[0]['boxes'].detach().cpu().numpy())] # Bounding boxes
    pred_score1 = list(pred[0]['scores'].detach().cpu().numpy())
    pred_labels1 = list(pred[0]['labels'].detach().cpu().numpy())

    pred_class = []
    pred_boxes = []
    pred_score = []
    pred_labels = []

    for i in range(len(pred_class1)):
        if(pred_class1[i] == 'car'):
            pred_class.append(pred_class1[i])
            pred_boxes.append(pred_boxes1[i])
            pred_score.append(pred_score1[i])
            pred_labels.append(pred_labels1[i])

    bounding_boxes = []
    for i in range(len(pred_labels)):
        if(pred_score[i] > 0.99):
            # for i in indices:
            bounding_boxes.append(pred_boxes[i])

    if(len(bounding_boxes) != 0):
        bounding_boxes = np.array(bounding_boxes)
        bounding_boxes = bounding_boxes.reshape([bounding_boxes.shape[0], 4])
        tracks = object_tracker.update(bounding_boxes)

        for i in range(tracks.shape[0]):
            bounding_boxes_ids = np.append(bounding_boxes_ids, np.array([tracks[i][4]]))
        
        count = np.unique(bounding_boxes_ids).shape[0]
        temp = count

        for i in range(bounding_boxes.shape[0]):
            cv2.rectangle(frame, (int(bounding_boxes[i][0]), int(bounding_boxes[i][1])), (int(bounding_boxes[i][2]), int(bounding_boxes[i][3])), (0, 255, 0), 2)
            cv2.putText(frame, "Car", (int(bounding_boxes[i][0])-10, int(bounding_boxes[i][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.putText(frame, f'Count: {int(count)}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)    

            cv2.imshow('result', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()