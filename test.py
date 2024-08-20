

import json
import os
from typing import List, Tuple
import numpy as np


def uv_to_coco_segmentation(uv: List[List[float]], offset: int = 0) -> List[float]:
    if not (len(uv) == 2 and len(uv[0]) == len(uv[1])):
        raise ValueError("The UV list must contain two lists of equal length.")

    # Convert UV coordinates to point tuples
    points = [(int(u), int(v)) for u, v in zip(uv[0], uv[1])]

    def offset_points(points, offset):
        offset_points_left = []
        offset_points_right = []
        for i in range(len(points) - 1):
            dx = points[i + 1][0] - points[i][0]
            dy = points[i + 1][1] - points[i][1]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx /= length
                dy /= length
                offset_dx = -dy * offset
                offset_dy = dx * offset
                offset_points_left.append((points[i][0] + offset_dx, points[i][1] + offset_dy))
                offset_points_right.append((points[i][0] - offset_dx, points[i][1] - offset_dy))
        return offset_points_left, offset_points_right

    if offset > 0:
        offset_left, offset_right = offset_points(points, offset)
        # Flatten the list of tuples for COCO format
        coco_segmentation = [coord for pair in offset_left + offset_right[::-1] for coord in pair]
    else:
        # If no offset is specified, return the original segmentation points interlaced
        coco_segmentation = [coord for coords in zip(uv[0], uv[1]) for coord in coords]

    return coco_segmentation


# Douglas-Peucker Algorithm to reduce the number of points
def douglas_peucker(points: List[Tuple[int, int]], epsilon: float) -> List[Tuple[int, int]]:
    # Define a function to calculate the perpendicular distance
    def perp_distance(point, start, end):
        if start == end:
            return np.linalg.norm(np.array(point) - np.array(start))
        return np.abs(np.cross(np.array(end) - np.array(start), np.array(start) - np.array(point)) / np.linalg.norm(np.array(end) - np.array(start)))

    # Find the point with the maximum distance from the line
    max_distance = 0
    max_index = 0
    for i in range(1, len(points) - 1):
        distance = perp_distance(points[i], points[0], points[-1])
        if distance > max_distance:
            max_distance = distance
            max_index = i

    # If max distance is greater than epsilon, recursively simplify
    if max_distance > epsilon:
        first_half = douglas_peucker(points[:max_index+1], epsilon)
        second_half = douglas_peucker(points[max_index:], epsilon)
        return first_half[:-1] + second_half
    else:
        return [points[0], points[-1]]

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

        # Specifying the categories and the offset
        if category_id in [1, 2, 7, 8]:  # All single
            offset = 15
        elif category_id in [3, 4, 9, 10]:  # Double
            offset = 20
        elif category_id in [5, 6, 11, 12]:  # Ld_Rs
            offset = 25        
        else:  # Default case
            offset = 0

        uv = lane_line["uv"]
        coco_annotation = {
            "id": lane_line_id_counter,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [uv_to_coco_segmentation(uv, offset)],
            "area": calculate_area(uv),
            "bbox": calculate_bbox(uv),
            "iscrowd": 0
        }
        coco_annotations.append(coco_annotation)
        lane_line_id_counter += 1

    modified_file_name =  os.path.basename(openlane_annotation["file_path"])

    # Include fixed image dimensions
    image_width, image_height = 1920, 1280

    return ({
        "images": [{"id": image_id, "file_name": modified_file_name, "width": image_width, "height": image_height}],
        "annotations": coco_annotations,
    }, lane_line_id_counter)


# Directory containing OpenLane v1 JSON files
folder_path = r'D:\OpenLane\images\test\detetcron_images\validation2\annoatations'
coco_output_path = r'D:\OpenLane\images\test\detetcron_images\validation2\images\labels.json'
#no_annotation_list_path = r'D:\porsche_ann_coco\no_annotations_list_validation.txt'

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
        print(file_path)
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