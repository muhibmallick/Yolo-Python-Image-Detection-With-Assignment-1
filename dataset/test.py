import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the YOLOv5 model from the ultralytics repository
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load and display the image
image_path = r'C:\Python\ML_Assignment\dataset\Image_Yolo_detect.png'
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# Convert the image to RGB format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform inference using the YOLOv5 model
results = model(image_rgb)

# Parse the results
detections = results.xyxy[0].cpu().numpy()  # Extract detections from results

# Print detections for debugging
print("Detections:", detections)

# Draw bounding boxes on the image
for detection in detections:
    x1, y1, x2, y2, conf, cls = detection
    cls = int(cls)
    label = f'{model.names[cls]}: {conf:.2f}'
    cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(image_rgb, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Convert the image back to BGR format for OpenCV display
image_bgr_with_boxes = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

# Save the image with detections to verify in case display doesn't work
output_path = r'C:\Python\ML_Assignment\dataset\Image_Yolo_detect_output.png'
cv2.imwrite(output_path, image_bgr_with_boxes)

# Display the image with detections
plt.imshow(image_rgb)
plt.title('YOLOv5 Detection Results')
plt.axis('off')
plt.show()
