import json
from typing import List, Tuple

# Function to convert UV coordinates to COCO segmentation format
def uv_to_coco_segmentation(uv: List[List[float]]) -> List[float]:
    return [coord for pair in zip(uv[0], uv[1]) for coord in pair]

# Function to calculate the bounding box from UV coordinates
def calculate_bbox(uv: List[List[float]]) -> Tuple[float, float, float, float]:
    min_x, max_x = min(uv[0]), max(uv[0])
    min_y, max_y = min(uv[1]), max(uv[1])
    width, height = max_x - min_x, max_y - min_y
    return min_x, min_y, width, height

# Function to calculate the approximate area of the lane line
def calculate_area(uv: List[List[float]]) -> float:
    average_width = 0.1  # Example width in meters or relevant units
    length = max(max(uv[0]) - min(uv[0]), max(uv[1]) - min(uv[1]))
    return length * average_width

# Function to convert OpenLane v1 annotation to COCO format
def convert_openlane_to_coco(openlane_annotation: dict) -> dict:
    coco_annotations = []
    for lane_line in openlane_annotation["lane_lines"]:
        coco_annotation = {
            "id": lane_line["track_id"],
            "image_id": 1,  # Assuming single image; replace with actual image ID mapping
            "category_id": lane_line["category"],
            "segmentation": [uv_to_coco_segmentation(lane_line["uv"])],
            "area": calculate_area(lane_line["uv"]),
            "bbox": calculate_bbox(lane_line["uv"]),
            "iscrowd": 0  # Assuming individual objects (lane lines are not crowd)
        }
        coco_annotations.append(coco_annotation)

    return {
        "images": [{"id": 1, "file_name": openlane_annotation["file_path"]}],
        "annotations": coco_annotations,
        "categories": [{"id": category, "name": str(category)} for category in range(1, 22)]  # Example category mapping
    }

# Path to the OpenLane v1 JSON file
openlane_file_path = r'D:\class test\yolo_test_an\2.json'
coco_output_path = r'D:\class test\yolo_test_an\path_to_output_coco_file.json'

# Reading the OpenLane v1 JSON file
with open(openlane_file_path, 'r') as file:
    openlane_data = json.load(file)

# Converting to COCO format
coco_data = convert_openlane_to_coco(openlane_data)

# Writing the output to a COCO format JSON file
with open(coco_output_path, 'w') as file:
    json.dump(coco_data, file, indent=4)

print(f"COCO format JSON file saved to {coco_output_path}")

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Function to draw the segmentation on the image
def draw_segmentation(image, segmentation, color=(0, 255, 0), thickness=2):
    segmentation = np.array(segmentation).reshape(-1, 2).astype(int)
    for i in range(1, len(segmentation)):
        cv2.line(image, tuple(segmentation[i - 1]), tuple(segmentation[i]), color, thickness)
    return image

# Function to draw the bounding box on the image
def draw_bbox(image, bbox, color=(255, 0, 0), thickness=2):
    x, y, w, h = map(int, bbox)
    return cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

# Load the COCO format JSON file
with open(coco_output_path, 'r') as file:
    coco_data = json.load(file)

# Assuming the image is in the same directory as the JSON file
image_path = r'D:\class test\yolo_test_an\2.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Draw each lane line segmentation and bounding box
for annotation in coco_data['annotations']:
    segmentation = annotation['segmentation'][0]  # Assuming each annotation has one segmentation
    bbox = annotation['bbox']
    
    # Draw the segmentation
    image = draw_segmentation(image, segmentation)
    
    # Draw the bounding box
    image = draw_bbox(image, bbox)

# Display the image with segmentations and bounding boxes
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.show()

