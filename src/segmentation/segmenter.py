"""
Wrapper class to streamline the use of SAM2AutomaticMaskGenerator:
    - Parametrization and creation
    - Automatic mask generation
    - Filtering 
    - Visualizing results
"""

import torch
from . import utils
import matplotlib.pyplot as plt
import cv2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
import copy
import numpy as np
from .well_mask import WellMask, create_from_sam2_mask
from .mask_container import MaskContainer
from cellpose import models, core, io, plot
from . import filtering

from analysis.utils import timeit

class Segmenter:
    """
    Attributes:
        self.model: the model wrapper to be used for segmentation. Must support a generate_masks(image) function.
        self.mask_generator: the SAM2AutomaticMaskGenerator instance.
        self.image: the input image for mask generation.
        self.masks [list(WellMask)]: the generated masks from the SAM2AutomaticMaskGenerator (filtered).
        self.masks_large: the generated masks categorized as large wells.
        self.masks_medium: the generated masks categorized as medium wells.
        self.masks_small: the generated masks categorized as small wells.
        self.masks_filtered_circularity: the masks filtered out based on circularity.
        self.masks_filtered_area: the masks filtered out based on area.
        self.masks_filtered_duplicate: the masks filtered out based on overlap.
    """
    def __init__(self, model: "SAM2Wrapper" or "CellposeSAMWrapper", filter_config):
        self.model = model

        self.masks = None
        self.large_masks = None
        self.medium_masks = None
        self.small_masks = None

        self.filter_config = filter_config

    def reset(self) -> "Segmenter":
        return Segmenter(model=self.model, filter_config=self.filter_config) # Wipe all class attributes 

    def set_image(self, image: np.ndarray):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = img

    def plot_image(self):
        if self.image is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(self.image)
            plt.axis('off')
            plt.show()
        else:
            print("Image must be set first.")

    @timeit
    def generate_masks(self, image: np.ndarray):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = img
        self.masks = self.model.generate_masks(img)
        return self.masks

    @timeit
    def filter_masks(self) -> None:
        self._filter_masks(
            circularity_threshold=self.filter_config["circularity_threshold"], 
            border_thres=self.filter_config["border_thres"], 
            max_groups=self.filter_config["max_groups"],
            eps_area=self.filter_config["eps_area"],
            eps_coords_medium=self.filter_config["eps_coords_medium"],
            eps_coords_small=self.filter_config["eps_coords_small"],
            eps_coords_small_exclude=self.filter_config["eps_coords_small_exclude"],
            rough_only=self.filter_config["rough_only"],
            debug=self.filter_config["debug"]
        )

    def _filter_masks(
        self, 
        circularity_threshold: float = 0.75, 
        border_thres: int = 10, 
        max_groups: int = None,
        eps_area: int = 60,
        eps_coords_medium: int = 75,
        eps_coords_small: int = 65,
        eps_coords_small_exclude: int = 25,
        rough_only: bool = False,
        debug: bool = False
    ) -> None:
        if len(self.masks) < 1:
            print("==== Warning: no masks detected! ====")
            return
            
        height, width = self.image.shape[:2]
        total_pixels = height * width

        # Rough filters
        self.masks._filter_by_circularity(circularity_threshold, debug=debug) # rough circularity filter
        self.masks._filter_by_area(total_pixels, debug=debug) # rough area filter
        self.masks._filter_duplicate_masks(debug=debug)  # SAM2AMG already filters using NMS, but we do it again since NMS threshold is difficult to tweak and we can exploit that no masks should overlap
        self.masks._filter_border_masks(border_thres=border_thres, debug=debug)

        if rough_only: # useful if it is unknown which chip part is being analyzed 
            return

        if len(self.masks) < 5:
            print(f"==== Warning: not enough masks to group by well size! Masks after rough filters: {len(self.masks)} ====")
            return
            
        # Group by well size - if max_groups is not provided, we test for 3 distinct groups, else we use 2
        if max_groups is None:
            # How to check which chip part: try k=3, if no difference between large and medium, or medium and small, do k=2
            area_ranges = utils.group_wells_by_area_kmeans(self.masks, max_groups=3)
            if area_ranges == {}:
                print("==== Warning: well grouping by area failed. ====")
                return
    
            upper_large = area_ranges["large"]["max"] 
            lower_large = area_ranges["large"]["min"]
            avg_area_large = (upper_large + lower_large) / 2
        
            upper_medium = area_ranges["medium"]["max"]
            lower_medium = area_ranges["medium"]["min"]
            avg_area_medium = (upper_medium + lower_medium) / 2
    
            upper_small = area_ranges["small"]["max"]
            lower_small = area_ranges["small"]["min"]
            avg_area_small = (upper_small + lower_small) / 2
    
            if avg_area_large > 2 * avg_area_medium and avg_area_medium > 2 * avg_area_small: # difference is large, therefore there are true large wells ;; normally the difference should be 2^2=4, but due to perspective distortion and segmentation error we use 2 to achieve better average results
                self.large_masks = [mask for mask in self.masks.masks if lower_large <= mask.area <= upper_large]
                self.large_masks = MaskContainer(image=self.image, masks=self.large_masks)
                self.medium_masks = [mask for mask in self.masks.masks if lower_medium <= mask.area <= upper_medium]
                self.medium_masks = MaskContainer(image=self.image, masks=self.medium_masks)
                upper_small = area_ranges["small"]["max"]
                lower_small = area_ranges["small"]["min"]
                self.small_masks = [mask for mask in self.masks.masks if lower_small <= mask.area <= upper_small]
                self.small_masks = MaskContainer(image=self.image, masks=self.small_masks)
                
            else: # difference is small, therefore there are probably no large wells (Chip part 2)
                area_ranges = utils.group_wells_by_area_kmeans(self.masks, max_groups=2)
                if area_ranges == {}:
                    print("==== Warning: well grouping by area failed. ====")
                    return
                upper_medium = area_ranges["medium"]["max"]
                lower_medium = area_ranges["medium"]["min"]
                self.medium_masks = [mask for mask in self.masks.masks if lower_medium <= mask.area <= upper_medium]
                self.medium_masks = MaskContainer(image=self.image, masks=self.medium_masks)
                upper_small = area_ranges["small"]["max"]
                lower_small = area_ranges["small"]["min"]
                self.small_masks = [mask for mask in self.masks.masks if lower_small <= mask.area <= upper_small]
                self.small_masks = MaskContainer(image=self.image, masks=self.small_masks)
        else:
            area_ranges = utils.group_wells_by_area_kmeans(self.masks, max_groups=max_groups)
            if max_groups == 3:
                upper_large = area_ranges["large"]["max"] 
                lower_large = area_ranges["large"]["min"]            
                self.large_masks = [mask for mask in self.masks.masks if lower_large <= mask.area <= upper_large]
                self.large_masks = MaskContainer(image=self.image, masks=self.large_masks)
                upper_medium = area_ranges["medium"]["max"]
                lower_medium = area_ranges["medium"]["min"]
                self.medium_masks = [mask for mask in self.masks.masks if lower_medium <= mask.area <= upper_medium]
                self.medium_masks = MaskContainer(image=self.image, masks=self.medium_masks)
                upper_small = area_ranges["small"]["max"]
                lower_small = area_ranges["small"]["min"]
                self.small_masks = [mask for mask in self.masks.masks if lower_small <= mask.area <= upper_small]
                self.small_masks = MaskContainer(image=self.image, masks=self.small_masks)
            elif max_groups == 2:
                self.large_masks = MaskContainer(image=self.image, masks=[]) # Necessary to avoid breaking analysis in this case
                upper_medium = area_ranges["medium"]["max"]
                lower_medium = area_ranges["medium"]["min"]
                self.medium_masks = [mask for mask in self.masks.masks if lower_medium <= mask.area <= upper_medium]
                self.medium_masks = MaskContainer(image=self.image, masks=self.medium_masks)
                upper_small = area_ranges["small"]["max"]
                lower_small = area_ranges["small"]["min"]
                self.small_masks = [mask for mask in self.masks.masks if lower_small <= mask.area <= upper_small]
                self.small_masks = MaskContainer(image=self.image, masks=self.small_masks)
 
        # Individual group filters
        all_masks = []

        # Large well filter: use a 1sigma intensity filter
        if self.large_masks is not None:
            if len(self.large_masks) > 3:
                intensities = []
                for mask in self.large_masks.masks:
                    intensity = np.mean(self.image[mask.segmentation.astype(bool)])
                    intensities.append(intensity)
                valid_indices = utils.sigma_filter(intensities, k=1)
                valid_masks = [self.large_masks.masks[i] for i in valid_indices]
                self.large_masks.masks = valid_masks
                all_masks += valid_masks
            else:
                all_masks += self.large_masks.masks

        # Medium well filter:
        if self.medium_masks is not None:
            mc = filtering.cluster_coords(self.medium_masks, eps=eps_coords_medium, debug=debug)
            self.medium_masks = mc
            all_masks += mc.masks

        # Small well filter:
        if self.small_masks is not None:
            mc = filtering.cluster_coords(self.small_masks, eps=eps_coords_small_exclude, debug=debug) # Exclude wells that are grouped too tightly (like the micro structure for distortion measurements)
            centroids_to_remove = [m.center for m in mc.masks]
            self.small_masks.masks = [mask for mask in self.small_masks.masks if mask.center not in centroids_to_remove]
            large_mc, other_mcs = filtering.cluster_area(self.small_masks, eps=eps_area, min_samples=10, debug=debug)
            mc = filtering.cluster_coords(large_mc, eps=eps_coords_small, debug=debug)
            self.small_masks = mc
            all_masks += mc.masks

        self.masks = MaskContainer(image=self.masks.image, masks=all_masks)

    @timeit
    def visualize_masks(self, mc: "MaskContainer" = None, title: str = "Image with Masks", identify: bool = False):
        _, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(self.image)
        ax[0].set_title("Raw Image")
        
        ax[1].imshow(self.image)
        ax[1].set_title(title)
        
        if mc is None:
            mc = self.masks
        mc.plot_all_wells(ax[1], identify=identify)
        
        for a in ax:
            a.axis('off')
        plt.tight_layout()
        plt.show()

    @timeit
    def visualize_masks_by_well_size(
            self, 
            large_masks: "MaskContainer" = None, 
            medium_masks: "MaskContainer" = None, 
            small_masks: "MaskContainer" = None, 
            title: str = "Image with Masks by Size", 
            identify: bool = False
        ):
        _, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(self.image, cmap='gray')
        ax[0].set_title("Raw Image")
        
        ax[1].imshow(self.image)
        ax[1].set_title(title)
        
        if large_masks is None:
            large_masks = self.large_masks
        if medium_masks is None:
            medium_masks = self.medium_masks
        if small_masks is None:
            small_masks = self.small_masks

        if large_masks != None and len(large_masks) > 0:
            large_masks.plot_all_wells(ax=ax[1], color=(200/255, 160/255, 255/255, 0.4), identify=identify) # light purple
        if medium_masks != None and len(medium_masks) > 0:
            medium_masks.plot_all_wells(ax=ax[1], color=(0/255, 255/255, 255/255, 0.4), identify=identify) # cyan
        if small_masks != None and len(small_masks) > 0:
            small_masks.plot_all_wells(ax=ax[1], color=(180/255, 140/255, 0/255, 0.4), identify=identify)  # dark yellow
        
        for a in ax:
            a.axis('off')
        plt.tight_layout()
        plt.show()
        
    def overlay_masks(self, masks: "MaskContainer", title: str = "Image with overlaid Masks"):
        _, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(self.image)
        ax[0].set_title("Raw Image")
        
        ax[1].imshow(self.image)
        ax[1].set_title(title)

        masks.plot_all_wells(ax=ax[1], color=(1,0,0,0.4), identify=False)  # gt masks in red
        self.masks.plot_all_wells(ax=ax[1], color=(0,0,1,0.4) , identify=False)  # pred masks in blue
        
        for a in ax:
            a.axis('off')
        plt.tight_layout()
        plt.show()


class SAM2Wrapper:
    def __init__(self, config = None, checkpoint = None, points_per_side: int = 128):
        """
        Initializes the SAM2 model and predictor.
        
        Parameters:
        -----------
        config : .yaml SAM2 configuration file.
        checkpoint : .pt SAM2 checkpoint file.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.config = config
        self.checkpoint = checkpoint
        self.model = build_sam2(config, checkpoint, device=device)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=points_per_side
        )

        self.predictor = None

    def generate_masks(self, image: np.ndarray):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(img)
        well_masks = []
        
        for mask in masks:
            well_mask = create_from_sam2_mask(img, mask)
            well_masks.append(well_mask)

        return MaskContainer(image=img, masks=well_masks)

    def generate_masks_with_prompts(self, image: np.ndarray, prompts: np.ndarray, labels: np.ndarray) -> MaskContainer:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if not self.predictor:
            self.predictor = SAM2ImagePredictor(self.model)
            
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            point_coords=prompts,
            point_labels=labels,
            multimask_output=False,
        )
        return MaskContainer(
            image=image,
            masks=[WellMask(segmentation=mask[0]) for mask in masks]
        )
        


class CellposeSAMWrapper:
    def __init__(self):
        if core.use_gpu() == False:
          raise ImportError("No GPU access, change your runtime")
        
        self.model = models.CellposeModel(gpu=True)

    def generate_masks(self, image: np.ndarray, flow_threshold: float = 0.4, cellprob_threshold: float = 0.0, tile_norm_blocksize: int = 0):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        masks, flows, styles = self.model.eval(
            gray_image,
            batch_size=32,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            normalize={"tile_norm_blocksize": tile_norm_blocksize}
        )
        well_masks=[]
        num_objects = masks.max()
        binary_masks = [(masks == i).astype(bool) for i in range(1, num_objects + 1)]
        for mask in binary_masks:
            well_mask = WellMask(segmentation=mask)
            well_masks.append(well_mask)
        return MaskContainer(masks=well_masks, image=gray_image)


class CellposeSAMSAM:
    def __init__(self, sam: SAM2Wrapper) -> None:
        self.cellpose = CellposeSAMWrapper()
        self.sam = sam

    def generate_masks(self, image):
        mc = self.cellpose.generate_masks(image)

        plt.figure(figsize=(5,5))
        plt.imshow(image, cmap="gray")
        mc.plot_all_wells()
        
        prompts = np.array([(x, y) for (y, x) in [m.center for m in mc.masks]]).reshape(-1, 1, 2)
        labels = np.ones([len(mc), 1])
        plt.scatter([m.center[1] for m in mc.masks], [m.center[0] for m in mc.masks])
        plt.show()
        
        mc_imp = self.sam.generate_masks_with_prompts(image=image, prompts=prompts, labels=labels)
        plt.figure(figsize=(5,5))
        plt.imshow(image, cmap="gray")
        mc_imp.plot_all_wells()

        return mc_imp







