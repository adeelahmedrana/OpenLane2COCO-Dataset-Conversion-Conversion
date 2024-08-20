# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 21:29:04 2023

@author: Adeel
"""

import json
import os
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
def convert_openlane_to_coco(openlane_annotation: dict, image_id: str, lane_line_id_counter: int) -> Tuple[dict, int]:
    coco_annotations = []
    for lane_line in openlane_annotation["lane_lines"]:
        category_id = lane_line["category"]

        # Skip lane lines with categories 20 and 21
        if category_id in [0, 20, 21]:
            continue

        uv = lane_line["uv"]
        coco_annotation = {
            "id": lane_line_id_counter,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [uv_to_coco_segmentation(uv)],
            "area": calculate_area(uv),
            "bbox": calculate_bbox(uv),
            "iscrowd": 0
        }
        coco_annotations.append(coco_annotation)
        lane_line_id_counter += 1

    modified_file_name = "/training/" + os.path.basename(openlane_annotation["file_path"])

    return ({
        "images": [{"id": image_id, "file_name": modified_file_name}],
        "annotations": coco_annotations,
    }, lane_line_id_counter)

# Directory containing OpenLane v1 JSON files
folder_path = r'D:\OpenLane\annoatations_v2\lane3d_1000\same\training'
coco_output_path = r'D:\OpenLane\annoatations_v2\coco\validation\coco_training.json'
no_annotation_list_path = r'D:\OpenLane\annoatations_v2\coco\training\no_annotations_list.txt'

# Initialize the overall COCO data structure with categories from 0 to 12
coco_data = {
    "images": [],
    "annotations": [],
    "categories": [{"id": category, "name": str(category)} for category in range(1, 13)]
}

lane_line_id_counter = 1  # Initialize the lane line ID counter
no_annotation_images = []  # List to hold names of images without annotations

for file_name in os.listdir(folder_path):
    if file_name.endswith('.json'):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            openlane_data = json.load(file)

        image_id = os.path.splitext(file_name)[0]  # File name as image_id
        image_coco_data, lane_line_id_counter = convert_openlane_to_coco(openlane_data, image_id, lane_line_id_counter)

        if image_coco_data["annotations"]:
            coco_data["images"].extend(image_coco_data["images"])
            coco_data["annotations"].extend(image_coco_data["annotations"])
        else:
            print(f"No annotations found for {file_name}. Skipping image details.")
            no_annotation_images.append(file_name)

# Write the output to a COCO format JSON file
with open(coco_output_path, 'w') as file:
    json.dump(coco_data, file, indent=4)

# Write the list of images without annotations to a text file
with open(no_annotation_list_path, 'w') as file:
    for image_name in no_annotation_images:
        file.write(image_name + '\n')

print(f"COCO format JSON file saved to {coco_output_path}")
print(f"List of images without annotations saved to {no_annotation_list_path}")