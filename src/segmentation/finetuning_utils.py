import os
import json
import shutil
import numpy as np
from pycocotools import mask as mask_utils
import cv2
import matplotlib.pyplot as plt

def construct_masks_from_json(path):
    """
    Construct masks from the annotations in the JSON file.
    """
    data = []
    with open(path, 'r') as f:
        json_data = json.load(f)
        for json_item in json_data:
            img_name = "-".join(json_item["file_upload"].split("-")[1:])
            ellipses = json_item["annotations"][0]["result"]
            
            annotations = []
            
            for ellipse in ellipses:
                height = ellipse["original_height"]
                width = ellipse["original_width"]
                ellipse_type = ellipse["value"]["ellipselabels"][0].split(" ")[0]
                area, bbox, segmentation = ellipse_to_rle_mask(height, width, ellipse["value"], ellipse["image_rotation"])
                
                annotations.append({
                    "bbox": bbox,
                    "area": area,
                    "segmentation": {
                        "size": [height, width],
                        "counts": segmentation["counts"]
                    },
                    "type": ellipse_type # "large", "medium", "small" or "well" if not specified
                }) 
            data.append({
                "image": {
                    "image_id": img_name.split(".")[0],  # Remove .jpg extension
                    "width": width,
                    "height": height,
                    "file_name": img_name,
                    },
                "annotations": annotations
            }) 
    return data


def ellipse_to_rle_mask(height, width, ellipse, image_rotation):
    mask, area, bbox = draw_mask(height, width, ellipse, image_rotation)

    # Convert mask from 255 to 1
    binary_mask = np.where(mask > 0, 1, 0).astype(np.uint8)

    # Encode using COCO RLE (expects Fortran-style order)
    rle = mask_utils.encode(np.asfortranarray(binary_mask))

    # Convert 'counts' from bytes to UTF-8 string
    rle["counts"] = rle["counts"].decode("utf-8")

    return area, bbox, {
        "size": [height, width],
        "counts": rle["counts"]
    }


def draw_mask(height, width, ellipse, image_rotation):
    canvas = np.zeros((height, width), dtype=np.uint8)
    x = ellipse["x"] / 100 * width
    y = ellipse["y"] / 100 * height
    radius_x = ellipse["radiusX"] / 100 * width
    radius_y = ellipse["radiusY"] / 100 * height
    rotation = ellipse["rotation"]
    
    total_rotation = rotation + image_rotation

    center = (int(x), int(y))
    axes = (int(radius_x), int(radius_y))

    cv2.ellipse(canvas, center, axes, total_rotation, 0, 360, 255, -1)
    
    area = int(np.count_nonzero(canvas))
    
    coords = cv2.findNonZero(canvas)
    x, y, w, h = cv2.boundingRect(coords)
    bbox = [float(x), float(y), float(w), float(h)]

    return canvas, area, bbox


# This function is necessary due to export issues with label-studio: some of the image names have commas and decimals removed
def remove_commas_percentages(path, name):
    files = os.listdir(path)
    for file in files:
        if file.endswith(".jpg"):
            new_name = file.replace(",", "")
            new_name = new_name.replace("%", "")

            if new_name == name:
                return file
    print(f"New name: {new_name}, Looking for: {name}")