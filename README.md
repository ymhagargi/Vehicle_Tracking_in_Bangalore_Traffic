# Vehicle_Tracking_in_Bangalore_Traffic
The system takes input video, detects cars in each frame, and tracks them across the frames to count the number of unique vehicles.

- The code uses a pre-trained Faster R-CNN model from the `torchvision` package to detect cars in the input video frames.
- The detected car bounding boxes are then passed to the Deep SORT tracker to track them across the frames and count the number of unique vehicles.
- The Deep SORT algorithm uses a Kalman filter to estimate the vehicle position and velocity and an intersection over union (IOU) metric to match the detected vehicles across the frames.
- The vehicle count is displayed in the output video frames.

# Faster RCNN vs YOLO

YOLO (You Only Look Once) is a real-time object detection algorithm that detects objects in images by dividing the image into a grid of cells and predicting the bounding boxes and class probabilities for each cell. It uses a single convolutional neural network (CNN) to predict the bounding boxes and class probabilities, which makes it faster than traditional object detection algorithms.

Faster R-CNN uses a two-stage approach. The first stage generates object proposals, which are regions of an image that are likely to contain an object. The second stage uses a CNN to classify the proposed regions and refine the bounding boxes.

YOLO is much faster than Faster R-CNN and is better suited for real-time applications. However, if accuracy is more important, Faster R-CNN is a better choice.

# SORT vs DeepSORT

SORT performs online tracking by using a combination of detection and data association techniques. It tracks objects in each frame by predicting their locations based on their previous positions and then associating them with newly detected objects based on the distance between them and how similar they are to each other in terms of appearance.

DeepSORT is an extension of SORT that uses deep learning techniques to improve tracking accuracy and robustness. It incorporates a deep neural network to extract more robust and discriminative features for object detection and tracking, making it more effective in handling occlusions, clutter, and other challenging scenarios.

SORT can perform object tracking in real-time with decent accuracy, whereas DeepSORT uses deep learning techniques to improve tracking accuracy and performance in challenging scenarios.
