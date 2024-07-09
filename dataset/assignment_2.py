import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (list): Coordinates of the first bounding box [x1, y1, x2, y2].
        box2 (list): Coordinates of the second bounding box [x1, y1, x2, y2].

    Returns:
        float: IoU value, a measure of the overlap between two bounding boxes.
    """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    # Calculate the coordinates of the intersection rectangle
    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)

    # Calculate the area of the intersection rectangle
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Calculate the area of both the prediction and ground-truth rectangles
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    # Calculate the union area by using the formula: union(A,B) = A + B - inter(A,B)
    union_area = box1_area + box2_area - inter_area

    # Compute the IoU
    iou = inter_area / union_area
    return iou

# Function to compute Average Precision (AP)
def compute_ap(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Compute the Average Precision (AP) for a set of ground truth and predicted boxes.

    Args:
        gt_boxes (list): Ground truth bounding boxes and classes.
        pred_boxes (list): Predicted bounding boxes, confidence scores, and classes.
        iou_threshold (float): IoU threshold to consider a prediction as a true positive.

    Returns:
        float: Average Precision (AP) value.
    """
    gt_classes = [box[4] for box in gt_boxes]  # Extract ground truth classes
    pred_classes = [box[5] for box in pred_boxes]  # Extract predicted classes
    y_true, y_scores = [], []

    # Loop over each ground truth box
    for i, gt_box in enumerate(gt_boxes):
        max_iou = 0  # Initialize max IoU as 0
        max_score = 0  # Initialize max score as 0

        # Loop over each predicted box
        for j, pred_box in enumerate(pred_boxes):
            if gt_classes[i] == pred_classes[j]:  # Check if classes match
                iou = calculate_iou(gt_box[:4], pred_box[:4])  # Calculate IoU
                if iou > max_iou:  # Update max IoU and max score if current IoU is greater
                    max_iou = iou
                    max_score = pred_box[4]

        # Append True if IoU is greater than or equal to threshold, otherwise False
        y_true.append(max_iou >= iou_threshold)
        y_scores.append(max_score)

    if len(y_true) == 0 or len(y_scores) == 0:
        return 0.0  # Return 0 if no true labels or scores are present

    # Calculate average precision score
    return average_precision_score(y_true, y_scores)

# Load the YOLOv5 model from the ultralytics repository
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set confidence and NMS thresholds
model.conf = 0.25  # Lowered Confidence threshold to detect more objects
model.iou = 0.5  # NMS IoU threshold

# Load and preprocess the image
image_path = r'C:\Python\ML_Assignment\dataset\Image_Yolo_detect.png'
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# Convert the image to RGB format as YOLO model expects images in RGB format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform inference using the YOLOv5 model
results = model(image_rgb)

# Parse the results to extract the detections
detections = results.xyxy[0].cpu().numpy()  # Extract detections from results

# Print detections for debugging
print("Detections:", detections)

# Draw bounding boxes on the image
for detection in detections:
    x1, y1, x2, y2, conf, cls = detection  # Extract individual elements of a detection
    cls = int(cls)  # Class needs to be integer for indexing
    label = f'{model.names[cls]}: {conf:.2f}'  # Create label with class name and confidence score
    color = (255, 0, 0)  # Dark blue color for the boxes and text
    cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # Draw rectangle
    cv2.putText(image_rgb, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # Put label text

# Convert the image back to BGR format for OpenCV display
image_bgr_with_boxes = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

# Save the image with detections to verify in case display doesn't work
output_path = r'C:\Python\ML_Assignment\dataset\Image_Yolo_detect_output.png'
cv2.imwrite(output_path, image_bgr_with_boxes)

# Display the image with detections using matplotlib
plt.imshow(image_rgb)
plt.title('YOLOv5 Detection Results')
plt.axis('off')  # Hide axis
plt.show()

# Assuming ground truth and predicted boxes are available
# Replace with actual ground truth data
gt_boxes = [
    # Example format: [x1, y1, x2, y2, class_index]
    [50, 50, 100, 100, 0],  # Example box for a person
    [150, 150, 200, 200, 2],  # Example box for a car
    [250, 250, 300, 300, 1]   # Example box for a bicycle
]

# Calculate mAP if ground truth boxes are available
if gt_boxes and detections.tolist():
    pred_boxes = detections.tolist()
    mAP = compute_ap(gt_boxes, pred_boxes)  # Calculate mean Average Precision
    print(f'mAP: {mAP:.4f}')
else:
    print('No ground truth boxes or detections to calculate mAP.')
