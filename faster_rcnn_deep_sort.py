import torch
import torchvision
import cv2
from sort.sort import *
from deep_sort_realtime.deepsort_tracker import DeepSort

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)

model.eval()

object_tracker = DeepSort(max_iou_distance=1, max_age=200,nms_max_overlap = 5)

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

def detect_cars(img, model, threshold = 0.5):

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    img1 = transform(img).to(device)

    with torch.no_grad():
        pred = model([img1])

    detections = []

    pred_class1 = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())] 
    pred_boxes1 = [[i[0], i[1], i[2], i[3]] for i in list(pred[0]['boxes'].detach().cpu().numpy())] 
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

    for i in range(len(pred_labels)):
        if(pred_score[i] > threshold):
            cv2.rectangle(img, (int(pred_boxes[i][0]), int(pred_boxes[i][1])), (int(pred_boxes[i][2]), int(pred_boxes[i][3])), (255,0, 0), 2)
            cv2.putText(img, "Car", (int(pred_boxes[i][0])-10, int(pred_boxes[i][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0, 0), 2)
            detections.append((pred_boxes[i], pred_score[i]))
    
    return detections

def update_track(tracks, count, track_ids):
    for t in tracks:
        if not t.is_confirmed():
            continue
        t_id = t.track_id
        if(t_id not in track_ids):
            count += 1
            track_ids.append(t_id)

    return track_ids, count

cap = cv2.VideoCapture('vid1.mp4')


track_ids = []
count = 0 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_cars(frame, model, 0.99)
    tracks = object_tracker.update_tracks(detections, frame=frame)
    track_ids, count = update_track(tracks, count, track_ids)
    cv2.putText(frame, f'Count: 9', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)

    cv2.imshow('result', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()