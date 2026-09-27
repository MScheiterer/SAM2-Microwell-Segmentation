import json
import numpy as np
from pycocotools import mask as mask_utils

import os
import sys

from segmentation.segmenter import Segmenter, SAM2Wrapper, CellposeSAMWrapper
from segmentation.well_mask import WellMask
from segmentation.mask_container import MaskContainer

from PIL import Image
from itertools import combinations

import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm.notebook import tqdm
import copy
import time

from analysis.utils import timeit
from scipy.spatial import cKDTree

def filter_overlapping_masks(pred_mask_container, gt_mask_container, min_overlap_pixels: int = 50, default_type: str = "large") -> ["MaskContainer", "MaskContainer"]:
    non_overlapping = []
    overlapping = []
    
    seen_mask_types = []
    for mask in gt_mask_container.masks:
        if mask.metadata["type"] in seen_mask_types:
            continue
        else:
            seen_mask_types.append(mask.metadata["type"])
    
    for mask in pred_mask_container.masks:
        overlap = False
        for gt_mask in gt_mask_container.masks:
            overlap_pixels = np.sum(mask.segmentation & gt_mask.segmentation)
            if overlap_pixels > min_overlap_pixels:
                mask.metadata["type"] = gt_mask.metadata["type"]
                overlapping.append(mask)
                overlap = True
                break
        if not overlap:
            if len(pred_mask_container) < 3:
                mask.metadata["type"] = default_type
            else:
                mask.metadata["type"] = pred_mask_container.approx_well_size(mask=mask, num_well_types=len(seen_mask_types))
            non_overlapping.append(mask)
    image = pred_mask_container.image
    return MaskContainer(masks=overlapping, image=image), MaskContainer(masks=non_overlapping, image=image)


def convert_anns_to_wellmasks(anns):
    gt_masks = []
    for ann in anns:
        rle = ann['segmentation']
        mask_type = ann["type"]
        decoded = mask_utils.decode(rle).astype(bool)  # shape (H, W)
        gt_masks.append(WellMask(
            segmentation=decoded,
            metadata={'type': mask_type}
        ))
    return gt_masks
    

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def calculate_average_iou(gt_mc, pred_mc, min_overlap_pixels: int = 10, max_centroid_dist: float = 100):
    matched_gt = set()
    ious = []

    gt_centers = np.array([gt.center for gt in gt_mc.masks])
    pred_centers = np.array([pred.center for pred in pred_mc.masks])
    gt_tree = cKDTree(gt_centers)

    for pred_idx, pred in enumerate(pred_mc.masks):
        # Find candidate GT masks by centroid distance
        candidate_idxs = gt_tree.query_ball_point(pred.center, r=max_centroid_dist)
        best_iou = 0
        best_gt_idx = None

        for i in candidate_idxs:
            if i in matched_gt:
                continue
            gt = gt_mc.masks[i]
            # Fast skip if little/no overlap
            overlap_pixels = np.sum(pred.segmentation & gt.segmentation)
            if overlap_pixels < min_overlap_pixels:
                continue

            iou = compute_iou(gt.segmentation, pred.segmentation)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_gt_idx is not None:
            matched_gt.add(best_gt_idx)
        ious.append(best_iou)

    if not ious:
        return 0

    return sum(ious) / len(ious)


def calculate_F1(precision, recall):
    return 0 if precision == 0 or recall == 0 else 2 * (precision * recall) / (precision + recall)