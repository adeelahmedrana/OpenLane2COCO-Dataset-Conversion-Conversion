# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 18:05:14 2023

@author: Adeel
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Function to draw the segmentation on the image
def draw_segmentation(image, segmentation, color=(0, 255, 0), thickness=2):
    segmentation = np.array(segmentation).reshape(-1, 2).astype(int)
    for i in range(1, len(segmentation)):
        cv2.line(image, tuple(segmentation[i - 1]), tuple(segmentation[i]), color, thickness)
    return image

# Load the COCO format JSON file
with open(coco_output_path, 'r') as file:
    coco_data = json.load(file)

# Assuming the image is in the same directory as the JSON file
image_path = coco_data['images'][0]['file_name']
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Draw each lane line segmentation
for annotation in coco_data['annotations']:
    segmentation = annotation['segmentation'][0]  # Assuming each annotation has one segmentation
    image = draw_segmentation(image, segmentation)

# Display the image with segmentations
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.show()
