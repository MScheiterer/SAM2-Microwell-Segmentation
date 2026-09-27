"""
Contains helper functions for the SAM2AutomaticMaskGenerator Wrapper class.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans


# Function to calculate circularity of a contour
def compute_circularity(mask):
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0  # No valid contour found

    contour = max(contours, key=cv2.contourArea)  # Get the largest contour
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:  # Avoid division by zero
        return 0

    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return circularity


def remove_mask_by_index(mask_list, original_masks, index_to_remove):
    """
    Remove a mask from mask_list that corresponds to original_masks[index_to_remove].
    """
    mask_to_remove = original_masks[index_to_remove]
    
    for i, mask in enumerate(mask_list):
        # Compare by reference first (fastest)
        if mask is mask_to_remove:
            mask_list.pop(i)
            return True
        # Fallback to array comparison if needed
        elif np.array_equal(mask.segmentation, mask_to_remove.segmentation):
            mask_list.pop(i)
            return True
    
    return False


def group_wells_by_area_kmeans(mask_container, max_groups=3):
    """
    Group well areas using K-means clustering and refine each group using a 2σ filter.

    Parameters:
    - areas: list or array of well areas
    - max_groups: max number of size groups to detect

    Returns:
    - grouped_ranges: dict like {'large': {'min': ..., 'max': ..., 'count': ...}, ...}
    """
    areas = np.array([mask.area for mask in mask_container.masks]).reshape(-1, 1)
    if len(areas) < max_groups:
        return {}

    kmeans = KMeans(n_clusters=max_groups, random_state=42).fit(areas)
    labels = kmeans.labels_

    grouped = {}
    for group_id in range(max_groups):
        group_areas = areas[labels == group_id].flatten()
        if len(group_areas) > 0:
            grouped[group_id] = {
                "min": float(min(group_areas)),
                "max": float(max(group_areas)),
                "count": len(group_areas),
                "mean": float(np.mean(group_areas))
            }

    # Sort by group mean (descending) and rename: large, medium, small
    sorted_groups = sorted(grouped.items(), key=lambda x: x[1]["mean"], reverse=True)
    renamed = {}
    if max_groups == 3:
        size_labels = ["large", "medium", "small"]
    elif max_groups == 2:
        size_labels = ["medium", "small"]
    else:
        return None

    for i, (original_id, stats) in enumerate(sorted_groups):
        if i < len(size_labels):
            renamed[size_labels[i]] = {
                "min": stats["min"],
                "max": stats["max"],
                "count": stats["count"]
            }
    return renamed


def sigma_filter(data, k: int = 2):
    mean_val = np.mean(data)
    std_val = np.std(data)
    lower_thresh = mean_val - k * std_val
    upper_thresh = mean_val + k * std_val
    valid_indices = [i for i, item in enumerate(data)
                        if lower_thresh <= item <= upper_thresh]
    return valid_indices

