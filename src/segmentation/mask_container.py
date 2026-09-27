import numpy as np
import matplotlib.pyplot as plt
from . import utils
import copy
from matplotlib.patches import Circle, Ellipse
 

class MaskContainer:
    def __init__(self, masks: list["WellMask"], image: np.ndarray, metadata: dict = None) -> None:
        self.image = image
        self.masks = sorted(masks, key=lambda mask: mask.center[0]) # sort by vertical position in image to assign consistent mask ids
        self.metadata = metadata or {}
        self.use_approx = False  # switch flag

        for i, mask in enumerate(self.masks):
            mask.mask_id = i

    def __len__(self) -> int:
        return len(self.masks)

    def __bool__(self) -> bool:
        return len(self.masks) > 0

    def get_mask_by_index(self, index: int) -> "WellMask":
        return self.masks[index]

    def remove_masks_by_id(self, mask_ids: int) -> None:
        masks = []
        for mask in self.masks:
            if mask.mask_id not in mask_ids:
                masks.append(mask)
        self.masks = masks                    

    def plot_all_wells(self, ax = None, color = (200/255, 160/255, 255/255, 0.4), identify: bool = False) -> np.ndarray:
        if ax is None: 
            ax = plt.gca()
        ax.set_autoscale_on(False)
    
        img = np.ones((self.image.shape[0], self.image.shape[1], 4))
        img[:, :, 3] = 0
        for mask in self.masks:
            mask.visualize_mask(ax=ax, img=img, color=color, identify=identify)
    
        ax.imshow(img)

        return img

    def _filter_by_circularity(self, circularity_threshold: float = 0.75, debug: bool = False) -> "MaskContainer":
        self.masks = [mask for mask in self.masks if utils.compute_circularity(mask.segmentation) >= circularity_threshold]

        removed_mc = MaskContainer(image=self.image, masks=[mask for mask in self.masks if utils.compute_circularity(mask.segmentation) < circularity_threshold])
        
        if debug:
            print(f"==== removed {len(removed_mc)} masks with circularity < {circularity_threshold} ====")
            
        return removed_mc

    def _filter_by_area(self, total_pixels: int, debug: bool = False) -> "MaskContainer":
        self.masks = [mask for mask in self.masks if 0.0001 < mask.area / total_pixels < 0.025]

        removed_mc = MaskContainer(image=self.image, masks=[mask for mask in self.masks if not (0.0001 < mask.area / total_pixels < 0.025)])
        if debug:
            print(f"==== removed {len(removed_mc)} masks with 0.0001 < area (%) < 0.025 ====")
            
        return removed_mc

    def _filter_duplicate_masks(self, min_overlap_pixels: int = 50, debug: bool = False) -> None:
        import copy
        non_overlapping_masks = copy.copy(self.masks)
        removed_masks = []
        removed_count = 0
        removed_indices = set()
        
        # Precompute bounding boxes
        bboxes = []
        for m in self.masks:
            ys, xs = np.where(m.segmentation)
            if len(xs) == 0 or len(ys) == 0:
                bboxes.append((0, 0, 0, 0))
            else:
                bboxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
        
        for i in range(len(self.masks)):
            if i in removed_indices:
                continue
            m1 = self.masks[i].segmentation.astype(np.uint8)
            bbox1 = bboxes[i]
            for j in range(i + 1, len(self.masks)):
                if j in removed_indices:
                    continue
                bbox2 = bboxes[j]
                # Quick reject if bounding boxes do not overlap
                if bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or \
                   bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]:
                    continue
                m2 = self.masks[j].segmentation.astype(np.uint8)
                overlap = np.sum(m1 & m2)
                
                if overlap >= min_overlap_pixels:
                    if self.masks[i].predicted_iou >= self.masks[j].predicted_iou:
                        mask_to_remove_idx = j
                        mask_to_keep = self.masks[i]
                        mask_to_remove = self.masks[j]
                    else:
                        mask_to_remove_idx = i
                        mask_to_keep = self.masks[j]
                        mask_to_remove = self.masks[i]
                    
                    if utils.remove_mask_by_index(non_overlapping_masks, self.masks, mask_to_remove_idx):
                        removed_masks.append(mask_to_remove)
                        removed_indices.add(mask_to_remove_idx)
                        removed_count += 1
    
                        if debug:
                            print(f"Removed overlapping mask: {overlap} pixels overlap, "
                                f"kept IoU={mask_to_keep.predicted_iou:.3f}, "
                                f"removed IoU={mask_to_remove.predicted_iou:.3f}")
                    
                    if mask_to_remove_idx == i:
                        break
        if debug:
            print(f"==== removed {removed_count} overlapping masks ====")
        self.masks = non_overlapping_masks
        self.masks_filtered_duplicate = MaskContainer(masks=removed_masks, image=self.image)

    def _filter_border_masks(self, border_thres: int = 20, debug: bool = False) -> "MaskContainer":
        filtered_masks = []
        border_masks = []
        for mask in self.masks:
            if not (is_mask_near_border(mask, border_thres=border_thres)):
                filtered_masks.append(mask)
            else:
                border_masks.append(mask)
                
        if debug:
            print(f"==== removed {len(self.masks) - len(filtered_masks)} masks touching the border within {border_thres} pixels====")

        self.masks = filtered_masks
        
        return MaskContainer(image=self.image, masks=border_masks)

    def approx_well_size(self, mask: np.array, num_well_types: int = 3) -> str:
        """Given a mask, approximate which well type it is using clustering."""
        area_ranges = utils.group_wells_by_area_kmeans(self, max_groups=num_well_types)
        if area_ranges is None:
            return "medium"
            
        if num_well_types == 3:
            upper_large = area_ranges["large"]["max"] 
            lower_large = area_ranges["large"]["min"]
            if lower_large <= mask.area <= upper_large:
                return "large"
    
        upper_medium = area_ranges["medium"]["max"]
        lower_medium = area_ranges["medium"]["min"]
        upper_small = area_ranges["small"]["max"]
        lower_small = area_ranges["small"]["min"]

        if lower_medium <= mask.area <= upper_medium:
            return "medium"

        if lower_small <= mask.area <= upper_small:
            return "small"

    def use_approximated(self, flag: bool = True, method: str = "ellipse", shrink_factor: float = 0.6) -> None:
        """Toggle between original and approximated mask."""
        self.use_approx = flag
        for mask in self.masks:
            mask.use_approximated(flag=flag, method=method, shrink_factor=shrink_factor)

    def as_binary(self) -> np.ndarray:
        return np.logical_or.reduce([m.segmentation for m in self.masks])


def is_mask_near_border(mask: "WellMask", border_thres: int = 20) -> bool:    
    h, w = mask.segmentation.shape
    y, x = np.nonzero(mask.segmentation)
    
    if len(y) == 0:
        return False  # empty mask
    
    min_x, max_x = x.min(), x.max()
    min_y, max_y = y.min(), y.max()
    
    # Check distance from each border
    if (min_x < border_thres or min_y < border_thres or
        (w - 1 - max_x) < border_thres or (h - 1 - max_y) < border_thres):
        return True
    
    return False

    
