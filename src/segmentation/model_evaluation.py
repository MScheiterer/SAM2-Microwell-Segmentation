import json
import numpy as np
from pycocotools import mask as mask_utils

import os
import sys

from segmentation.segmenter import *
from segmentation.well_mask import WellMask
from segmentation.mask_container import MaskContainer
from segmentation.eval_utils import *

from PIL import Image
from itertools import combinations

import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm.notebook import tqdm
import copy
import time


def create_no_detection_row(image_name, gt_masks_all, gt_masks_large=None, gt_masks_medium=None, gt_masks_small=None):
    """
    Create a row for cases where no masks are detected by the model.
    Sets metrics to 0 if corresponding gt_masks > 0, else NaN.
    
    Args:
        image_name (str): Name of the image
        gt_masks_all (int): Number of ground truth masks (all sizes)
        gt_masks_large (int, optional): Number of large ground truth masks
        gt_masks_medium (int, optional): Number of medium ground truth masks  
        gt_masks_small (int, optional): Number of small ground truth masks
        
    Returns:
        dict: Row dictionary ready to append to DataFrame
    """
    
    # Helper function to set metric value
    def get_metric_value(gt_count):
        if gt_count is None or gt_count == 0:
            return np.nan
        else:
            return 0.0
    
    row = {
        "image_name": image_name,
        
        # SAM2 detected masks (all 0 since no detection)
        "sam2_masks_all": 0,
        "sam2_masks_large": 0 if gt_masks_large is not None else None,
        "sam2_masks_medium": 0 if gt_masks_medium is not None else None,
        "sam2_masks_small": 0 if gt_masks_small is not None else None,
        "sam2_masks_incorrect": 0,
        
        # Ground truth masks (preserve original values)
        "gt_masks_all": gt_masks_all,
        "gt_masks_large": gt_masks_large,
        "gt_masks_medium": gt_masks_medium,
        "gt_masks_small": gt_masks_small,
        
        # Segmentation time (NaN since no masks were processed)
        "segmentation_time_per_mask": np.nan,
        
        # Precision metrics (0 if gt exists, NaN otherwise)
        "precision_all": get_metric_value(gt_masks_all),
        "precision_large": get_metric_value(gt_masks_large),
        "precision_medium": get_metric_value(gt_masks_medium),
        "precision_small": get_metric_value(gt_masks_small),
        
        # Recall metrics (0 if gt exists, NaN otherwise) 
        "recall_all": get_metric_value(gt_masks_all),
        "recall_large": get_metric_value(gt_masks_large),
        "recall_medium": get_metric_value(gt_masks_medium),
        "recall_small": get_metric_value(gt_masks_small),
        
        # F1 metrics (0 if gt exists, NaN otherwise)
        "F1_all": get_metric_value(gt_masks_all),
        "F1_large": get_metric_value(gt_masks_large),
        "F1_medium": get_metric_value(gt_masks_medium),
        "F1_small": get_metric_value(gt_masks_small),
        
        # IoU metrics (0 if gt exists, NaN otherwise)
        "average_IoU_all": get_metric_value(gt_masks_all),
        "average_IoU_large": get_metric_value(gt_masks_large),
        "average_IoU_medium": get_metric_value(gt_masks_medium),
        "average_IoU_small": get_metric_value(gt_masks_small)
    }
    
    return row

def append_no_detection_row(df, image_name, gt_masks_all, gt_masks_large=None, gt_masks_medium=None, gt_masks_small=None):
    """
    Append a no-detection row to an existing DataFrame.
    
    Args:
        df (pd.DataFrame): Existing DataFrame to append to
        image_name (str): Name of the image
        gt_masks_all (int): Number of ground truth masks (all sizes)
        gt_masks_large (int, optional): Number of large ground truth masks
        gt_masks_medium (int, optional): Number of medium ground truth masks  
        gt_masks_small (int, optional): Number of small ground truth masks
        
    Returns:
        pd.DataFrame: DataFrame with the new row appended
    """
    no_detection_row = create_no_detection_row(
        image_name, gt_masks_all, gt_masks_large, gt_masks_medium, gt_masks_small
    )
    
    # Convert to DataFrame row and append
    new_row_df = pd.DataFrame([no_detection_row])
    result_df = pd.concat([df, new_row_df], ignore_index=True)
    
    return result_df


def calculate_metrics(
    mc_correct: "MaskContainer", 
    mc_incorrect: "MaskContainer", 
    gt_mc: "MaskContainer", 
    welltype: str = "all"
) -> [float, float, float, float, "MaskContainer", "MaskContainer"]:
    if welltype != "all":
        mc_correct = MaskContainer(
            masks=[mask for mask in mc_correct.masks if mask.metadata["type"] == welltype],
            image=mc_correct.image
        )
        mc_incorrect = MaskContainer(
            masks=[mask for mask in mc_incorrect.masks if mask.metadata["type"] == welltype],
            image=mc_correct.image
        )
    if mc_correct:
        mean_iou = calculate_average_iou(gt_mc, mc_correct)
        recall = float(len(mc_correct)/len(gt_mc)) 
        precision = len(mc_correct) / (len(mc_correct) + len(mc_incorrect))
        F1 = calculate_F1(precision, recall)
    elif len(gt_mc) < 1: # No masks to be found
        mean_iou = None
        recall = None
        precision = None
        F1 = None
    else: # no masks detected
        mean_iou = None
        recall = 0
        precision = 0
        F1 = 0

    return mean_iou, recall, precision, F1, mc_correct, mc_incorrect



def model_comparer(image_path, annotation_path, segmenter: Segmenter, experiment_dir, experiment_name):
    file_counter = 0
    sum_correct_masks = 0

    columns = [
        "image_name", 
        "sam2_masks_all", 
        "sam2_masks_large", 
        "sam2_masks_medium", 
        "sam2_masks_small", 
        "sam2_masks_incorrect",
        "precision_all",
        "precision_large",
        "precision_medium",
        "precision_small",
        "recall_all",
        "recall_large",
        "recall_medium",
        "recall_small",
        "F1_all",
        "F1_large",
        "F1_medium",
        "F1_small",
        "gt_masks_all", 
        "gt_masks_large",
        "gt_masks_medium",
        "gt_masks_small",
        "segmentation_time_per_mask",
        "average_IoU_all", 
        "average_IoU_large",
        "average_IoU_medium",
        "average_IoU_small",
    ]
    dtype_map = {
        "image_name": str,
        "sam2_masks_all": int,
        "sam2_masks_large": int,
        "sam2_masks_medium": int,
        "sam2_masks_small": int,
        "sam2_masks_incorrect": int,
        "precision_all": float,
        "precision_large": float,
        "precision_medium": float,
        "precision_small": float,
        "recall_all": float,
        "recall_large": float,
        "recall_medium": float,
        "recall_small": float,
        "F1_all": float,
        "F1_large": float,
        "F1_medium": float,
        "F1_small": float,
        "gt_masks_all": int,
        "gt_masks_large": int,
        "gt_masks_medium": int,
        "gt_masks_small": int,
        "segmentation_time_per_mask": float,
        "average_IoU_all": float,
        "average_IoU_large": float,
        "average_IoU_medium": float,
        "average_IoU_small": float,
    }
    
    df = pd.DataFrame(columns=columns).astype(dtype_map)

    for image_file in tqdm(os.listdir(image_path)):
        if image_file.endswith(".jpg"):
            image_base, _ = os.path.splitext(image_file)
            annotation_file = f"{image_base}.json"
            annotation_file_path = os.path.join(annotation_path, annotation_file)
    
            if os.path.isfile(annotation_file_path):
                print(f"Processing image: {image_file}")
                print(f"Opening annotation: {annotation_file_path}")
                with open(annotation_file_path, 'r') as f:
                    annotation_data = json.load(f)

                    img = np.array(Image.open(os.path.join(image_path, image_file)).convert("L"))
                    anns = annotation_data["annotations"]

                    gt_mc_all = MaskContainer(masks=convert_anns_to_wellmasks(anns), image=img)
                    gt_mc_large = MaskContainer(
                        masks=[mask for mask in gt_mc_all.masks if mask.metadata["type"]=="large"],
                        image=img
                    )
                    gt_mc_medium = MaskContainer(
                        masks=[mask for mask in gt_mc_all.masks if mask.metadata["type"]=="medium"],
                        image=img
                    )
                    gt_mc_small = MaskContainer(
                        masks=[mask for mask in gt_mc_all.masks if mask.metadata["type"]=="small"],
                        image=img
                    )

                    segmenter = segmenter.reset()
                    segmentation_start_time = time.perf_counter()
                    segmenter.generate_masks(img)
                    segmentation_end_time = time.perf_counter()
                    segmentation_exec_time = segmentation_end_time - segmentation_start_time

                    segmenter.visualize_masks()
                    segmenter.filter_masks()
                    segmenter.visualize_masks_by_well_size()

                    print("Num masks: ", len(segmenter.masks))
                    if len(segmenter.masks) < 4:
                        df = append_no_detection_row(
                            df=df,
                            image_name=image_file,
                            gt_masks_all=len(gt_mc_all),
                            gt_masks_large=len(gt_mc_large) if gt_mc_large.masks is not None else None,
                            gt_masks_medium=len(gt_mc_medium) if gt_mc_medium.masks is not None else None,
                            gt_masks_small=len(gt_mc_small) if gt_mc_small.masks is not None else None
                        )
                        continue

                    # Match predicted masks to ground truth to find true/false positives
                    mc_correct, mc_incorrect = filter_overlapping_masks(segmenter.masks, gt_mc_all)
                    segmenter.visualize_masks(mc_correct)
                    segmenter.overlay_masks(gt_mc_all)

                    mean_iou_all, recall_all, precision_all, F1_all, _, _ = calculate_metrics(mc_correct, mc_incorrect, gt_mc_all, welltype="all")
                    mean_iou_large, recall_large, precision_large, F1_large, mc_large_correct, mc_large_incorrect = calculate_metrics(mc_correct, mc_incorrect, gt_mc_large, welltype="large")
                    mean_iou_medium, recall_medium, precision_medium, F1_medium, mc_medium_correct, mc_medium_incorrect = calculate_metrics(mc_correct, mc_incorrect, gt_mc_medium, welltype="medium")
                    mean_iou_small, recall_small, precision_small, F1_small, mc_small_correct, mc_small_incorrect = calculate_metrics(mc_correct, mc_incorrect, gt_mc_small, welltype="small")

                    row = {
                        "image_name": image_file,
                        "sam2_masks_all": len(mc_correct), 
                        "sam2_masks_large": len(mc_large_correct) if mc_large_correct.masks is not None else None, 
                        "sam2_masks_medium": len(mc_medium_correct) if mc_medium_correct.masks is not None else None, 
                        "sam2_masks_small": len(mc_small_correct) if mc_small_correct.masks is not None else None, 
                        "sam2_masks_incorrect": len(mc_incorrect) if mc_incorrect.masks is not None else None,
                        "gt_masks_all": len(gt_mc_all), 
                        "gt_masks_large": len(gt_mc_large) if gt_mc_large.masks is not None else None,
                        "gt_masks_medium": len(gt_mc_medium) if gt_mc_medium.masks is not None else None,
                        "gt_masks_small": len(gt_mc_small) if gt_mc_small.masks is not None else None,
                        "segmentation_time_per_mask": segmentation_exec_time / (len(mc_correct) + len(mc_incorrect)) if len(mc_correct) > 0 or len(mc_incorrect) > 0 else None,
                        "precision_all": precision_all,
                        "precision_large": precision_large,
                        "precision_medium": precision_medium,
                        "precision_small": precision_small,
                        "recall_all": recall_all,
                        "recall_large": recall_large,
                        "recall_medium": recall_medium,
                        "recall_small": recall_small,
                        "F1_all": F1_all,
                        "F1_large": F1_large,
                        "F1_medium": F1_medium,
                        "F1_small": F1_small,
                        "average_IoU_all": mean_iou_all,
                        "average_IoU_large": mean_iou_large if mean_iou_large is not None else None,
                        "average_IoU_medium": mean_iou_medium if mean_iou_medium is not None else None,
                        "average_IoU_small": mean_iou_small if mean_iou_small is not None else None
                    }
                    df.loc[len(df)] = row

                    print(json.dumps(row, indent=4))
            else:
                print(f"Annotation for {image_file} not found.")
            file_counter += 1
            
    print(f"==== Average masks found: {sum_correct_masks/file_counter}% ====")
    print(f"==== {file_counter} files found. ====")

    average_masks_incorrect = df["sam2_masks_incorrect"].astype(float).mean()

    average_precision_all = df["precision_all"].astype(float).mean()
    average_precision_large = df["precision_large"].astype(float).mean()
    average_precision_medium = df["precision_medium"].astype(float).mean()
    average_precision_small = df["precision_small"].astype(float).mean()

    average_recall_all = df["recall_all"].astype(float).mean()
    average_recall_large = df["recall_large"].astype(float).mean()
    average_recall_medium = df["recall_medium"].astype(float).mean()
    average_recall_small = df["recall_small"].astype(float).mean()
    
    average_iou_all = df["average_IoU_all"].astype(float).mean()
    average_iou_large = df["average_IoU_large"].astype(float).mean()
    average_iou_medium = df["average_IoU_medium"].astype(float).mean()
    average_iou_small = df["average_IoU_small"].astype(float).mean()

    average_segmentation_time = df["segmentation_time_per_mask"].astype(float).mean()

    avg_F1_all = df["F1_all"].astype(float).mean()
    avg_F1_large = df["F1_large"].astype(float).mean()
    avg_F1_medium = df["F1_medium"].astype(float).mean()
    avg_F1_small = df["F1_small"].astype(float).mean()
    
    summary_row = {
        "image_name": "average",
        "sam2_masks_all": "", 
        "sam2_masks_large": "", 
        "sam2_masks_medium": "", 
        "sam2_masks_small": "", 
        "sam2_masks_incorrect": average_masks_incorrect,
        "gt_masks_all": "", 
        "gt_masks_large": "",
        "gt_masks_medium": "",
        "gt_masks_small": "",
        "segmentation_time_per_mask": average_segmentation_time,
        "precision_all": average_precision_all,
        "precision_large": average_precision_large,
        "precision_medium": average_precision_medium,
        "precision_small": average_precision_small,
        "recall_all": average_recall_all,
        "recall_large": average_recall_large,
        "recall_medium": average_recall_medium,
        "recall_small": average_recall_small,
        "F1_all": avg_F1_all,
        "F1_large": avg_F1_large,
        "F1_medium": avg_F1_medium,
        "F1_small": avg_F1_small,
        "average_IoU_all": average_iou_all,
        "average_IoU_large": average_iou_large,
        "average_IoU_medium": average_iou_medium,
        "average_IoU_small": average_iou_small
    }
    df.loc[len(df)] = summary_row
    df.to_csv(f"{experiment_dir}/{experiment_name}_summary.csv", index=False)
    return df