import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io, measure, morphology, color, img_as_float, img_as_ubyte
from skimage.measure import label, regionprops
from skimage.transform import resize
import imreg_dft as ird
from segmentation.eval_utils import calculate_average_iou
from segmentation.well_mask import WellMask
from segmentation.mask_container import MaskContainer
from segmentation.segmenter import Segmenter, CellposeSAMWrapper
from scipy.spatial import cKDTree
from analysis.utils import timeit

class Camera:
    def __init__(self, zoom: float, um_per_pixel: float):
        self.zoom = zoom
        self.um_per_pixel = um_per_pixel


class CameraCalibrator:
    def __init__(
        self, 
        setup_image: np.ndarray,  
        camera: "Camera",
        segmenter: "Segmenter",
        ref_images: list[np.ndarray],
        ref_radii: list[int],  
        transform: np.ndarray = None,
        calculate_individual: bool = False
    ) -> None:
        self.setup_image = setup_image

        self._resized = False
        self.setup_mask = None
        self.segmenter = segmenter

        self.ref_images = ref_images
        self.ref_image = None
        self.ref_radii = ref_radii
        self.ref_radius = None
        self.camera = camera

        self.calculate_individual = calculate_individual

        if transform is not None:
            assert transform.shape == (2, 3), f"Affine transformation matrix must have shape (2, 3), got {transform.shape}"
            self._transform = transform
        else:
            self._transform = None

        self._homography = None
        self._reverse_transform = None
            
        self.ref_mask = None
        self.chip_region = None


    def _initialize(self, image: np.ndarray, chip_region: int = 1, debug: bool = False) -> None:
        # Get reference mask
        self.segmenter.generate_masks(self.setup_image)
        self.segmenter.filter_masks()
        if chip_region == 1 or chip_region == 3:
            self.setup_mask = self.segmenter.large_masks.masks[0]
        elif chip_region == 2: 
            self.setup_mask = self.segmenter.medium_masks.masks[0]
        if debug:
            self.segmenter.visualize_masks_by_well_size()
            
        self.setup_mask.use_approximated(True, method="ellipse", shrink_factor=1)
        self.ref_radius = self.ref_radii[f'region{chip_region}']
        self.ref_image = self.ref_images[f'region{chip_region}'] 

        if self.ref_image.shape != self.setup_image.shape:
            # Resize schematic to image shape
            affine_transformed = self.apply_affine(image=self.setup_image)
            self.segmenter.generate_masks(affine_transformed)
            self.segmenter.filter_masks()

            if debug:
                self.segmenter.visualize_masks_by_well_size()
            mc = self.segmenter.masks
            resize_shape = calculate_resize(mc=mc, debug=debug)
            resized_schematic, _ = resize_uniform(self.ref_image, resize_shape)
            self.ref_image = pad_to_shape(resized_schematic, self.setup_image.shape)
            self.ref_images[f'region{chip_region}'] = self.ref_image

        self.chip_region = chip_region


    @property
    def affine_transform(self) -> np.ndarray:
        """
        Returns the affine transformation matrix used to align the setup image.
    
        If the transformation matrix is not yet computed, it calculates it using
        the setup image, and caches the components: scaling (S), rotation (R),
        and translation (T).
    
        Returns:
            np.ndarray: A 2×3 affine transformation matrix.
        """
        if self._transform is None:
            self._transform, S, R, T = self._calculate_affine_transform(self.setup_image)
            self.S = S # Scaling transformation
            self.R = R # Rotation
            self.T = T # Translation
        return self._transform

    @property
    def reverse_affine_transform(self) -> np.ndarray:
        """
        Returns the inverse of the affine transformation matrix.
    
        If the inverse is not yet computed, it calculates it using OpenCV’s
        invertAffineTransform function on the forward affine transform.
    
        Returns:
            np.ndarray: A 2×3 inverse affine transformation matrix.
        """
        if self._reverse_transform is None:
            self._reverse_transform = cv2.invertAffineTransform(self.affine_transform)
        return self._reverse_transform

    def apply_affine(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the affine transformation to a given image.
    
        This function performs a geometric transformation of the input image
        using nearest-neighbor interpolation (to preserve intensity values)
        and outputs an image of the same shape.
    
        Args:
            image (np.ndarray): The input image to transform.
    
        Returns:
            np.ndarray: The transformed image.
        """
        T = self.affine_transform
        output_size = (image.shape[1], image.shape[0])
        return cv2.warpAffine(image, T, dsize=output_size, flags=cv2.INTER_NEAREST) # INTER_NEAREST has a smaller effect on intensity compared to INTER_LINEAR

    def apply_reverse_affine(self, pts: np.ndarray, M: np.ndarray = None) -> np.ndarray:
        """
        Applies the inverse affine transformation to a set of 2D points.
    
        The input coordinates are assumed to be in (y, x) format and are converted
        to (x, y) before applying the inverse transformation. The result is then
        converted back to (y, x).
    
        Args:
            pts (np.ndarray): An array of shape (N, 2) containing (y, x) points.
    
        Returns:
            np.ndarray: An array of shape (N, 2) with the reverse-transformed (y, x) points.
        """
        if M is None:
            M = self.reverse_affine_transform
            
        # Convert (y, x) → (x, y)
        coords = np.array(pts)[:, [1, 0]]
        ones = np.ones((coords.shape[0], 1))
        hom_coords = np.hstack([coords, ones])  # shape: (N, 3)

        transformed = hom_coords @ M.T  # shape: (N, 2)

        # Convert back (x, y) → (y, x)
        transformed = transformed[:, [1, 0]]
        return transformed
        
    def _calculate_ref_circle(self):
        """
        Creates a binary circular reference mask within the setup image.
    
        The reference circle is centered on the `setup_mask.center` and has a radius 
        determined by converting the real-world `ref_radius` (in microns) to pixels, 
        using the camera's microns-per-pixel value and zoom factor.
    
        The resulting binary mask is stored as a `WellMask` in `self.ref_mask`, and 
        includes metadata containing the circle's center and pixel radius.
    
        This reference is later used to compute affine transformations for image alignment.
        """
        radius_px = int(self.ref_radius / (self.camera.um_per_pixel / self.camera.zoom))
        centroid = self.setup_mask.center
        
        circle_mask = np.zeros(self.setup_image.shape[:2], dtype=np.uint8)
        cv2.circle(circle_mask, centroid, radius_px, 255, -1)
        self.ref_mask = WellMask(segmentation=circle_mask, metadata={"center": centroid, "radius": radius_px})

    def _calculate_affine_transform(self, image: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the full affine transformation matrix to align an elliptical mask to a 
        circular reference mask.
    
        This method uses the ellipse approximation from `self.setup_mask` to calculate:
        - `S`: a scaling matrix to normalize the ellipse axes to match the circular reference
        - `R`: a rotation matrix to undo the ellipse’s angle
        - `T`: a translation matrix to recenter the image after scaling and rotation
    
        The matrices are combined (in the order: Scale → Rotate → Translate) into a single 
        affine transform `TRS`, which can be used to warp the image or reverse-transform points.
    
        Args:
            image (np.ndarray): The image to be used for transformation context (only shape is used).
    
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - `TRS`: Combined affine transform (2×3)
                - `S`: Scaling matrix (2×3)
                - `R`: Rotation matrix (2×3)
                - `T`: Translation matrix (2×3)
        """
        if self.ref_mask is None:
            self._calculate_ref_circle()
            
        # Get ellipse parameters
        center = self.setup_mask.approximation_info["center"]
        axes = self.setup_mask.approximation_info["axes"]
        angle_deg = self.setup_mask.approximation_info["angle"]
        
        # Unpack correctly: center is (row, col) = (y, x), but OpenCV uses (x, y)
        cy, cx = center
        w, h = axes  # widths (diameters), not radii
    
        # Ensure w is the major axis
        if h > w:
            w, h = h, w
            angle_deg += 90  # adjust angle if swapped
    
        # Convert to radii
        a = w / 2
        b = h / 2

        # Scaling (also 2x3 affine transformation)
        s = self.ref_mask.metadata["radius"] 
        S = np.zeros((2, 3))
        S[:, :2] = np.array([
            [s/a, 0],
            [0, s/b]
        ])
    
        # Rotation matrix to undo ellipse orientation (Affine transformation matrix 2x3)
        R = cv2.getRotationMatrix2D(center=center, angle=angle_deg-180, scale=1)
    
        # Translation (move center back to where it started, to keep whole image info)
        # here the scaling and rotating have to be taken into account when calculating the translation vector
        T = np.zeros((2, 3))
        RS = combine_affine_matrices(S, R)
        sr_center = [cx, cy] @ RS[:, :2] + RS[:, 2]
        sr_cx, sr_cy = sr_center
        T[:, :2] = np.eye(2)
        T[:, 2] = [cx - sr_cx, cy - sr_cy - 100] # Shift an extra 100px upwards (for image coordinates top left corner is 0,0) to center image. This should be adjusted if not using the top large well for reference

        RS = combine_affine_matrices(S, R)
        TRS = combine_affine_matrices(RS, T)
    
        return TRS, S, R, T

    @timeit
    def _apply_imreg_dft(self, image: np.ndarray, debug: bool = False) -> [np.ndarray, dict]:
        constraints = {
            'angle': [-30, 30],
        }

        match = ird.similarity(
            image, 
            self.ref_image, 
            numiter=3, 
            filter_pcorr=0,
            exponent='inf',
            constraints=constraints
        )
        if debug:
            print(f"Translation: {match['tvec']}")
            print(f"Scale: {match['scale']}")
            print(f"Rotation: {match['angle']}°")
    
        # Transform schematic to align with the image
        aligned_ref = ird.transform_img(
            img=self.ref_image, 
            scale=match['scale'], 
            angle=match['angle'], 
            tvec=match['tvec']
        )
        
        if debug:
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
            axes[0].imshow(image, cmap='gray')
            axes[0].set_title("Image")
            
            axes[1].imshow(image, cmap='gray')
            overlay = np.zeros((*aligned_ref.shape, 4))
            overlay[..., 2] = 1.0
            overlay[..., 3] = aligned_ref * 0.5
            
            axes[1].imshow(overlay)
            axes[1].set_title("Overlay: Image vs. Aligned Reference")
            
            for ax in axes:
                ax.axis("off")
            
            plt.tight_layout()
            plt.show()

        return aligned_ref, match

    def _match_centroids(self, src_image: np.ndarray, dst_image: np.ndarray, debug: bool = False) -> [np.ndarray, np.ndarray]:
        # Match mask pairs between approximated and schematic
        individual_masks = split_into_individual_masks(dst_image)
        gt_mc = MaskContainer(
            image=dst_image,
            masks=[WellMask(segmentation=mask) for mask in individual_masks]
        )
        self.segmenter.generate_masks(src_image)
        self.segmenter.filter_masks()
        mc = self.segmenter.masks
        
        # Find overlapping pairs and create corresponding centroid lists
        src_pts, dst_pts = find_overlapping_pairs(mc, gt_mc)
        
        if debug:
            print(f"Src: {len(src_pts)}, Dst: {len(dst_pts)}")

        return src_pts, dst_pts

    @property
    def homography(self):
        if self._homography is None:
            self._homography, _ = self.calculate_homography(
                image=self.setup_image,
                reverse=self.reverse,
                debug=self.debug
            )
        return self._homography

    @timeit
    def calculate_homography(
        self, 
        image: np.ndarray, 
        reverse: bool = True,
        use_binary: bool = True,
        debug: bool = False
    ) -> np.ndarray: 
         # Calculate affine transformation
        affine_transformed = self.apply_affine(image=image)

        # Align Schematic - use binary when aligning high RI image
        if use_binary:
            self.segmenter.generate_masks(affine_transformed)
            self.segmenter.filter_masks()
            binary = self.segmenter.masks.as_binary()
            aligned_schematic, match = self._apply_imreg_dft(image=binary, debug=debug)
        else:
            aligned_schematic, match = self._apply_imreg_dft(image=affine_transformed, debug=debug)

        # Calculate overlapping centroid pairs between aligned schematic and affine_transformed
        src_pts, dst_pts = self._match_centroids(src_image=affine_transformed, dst_image=aligned_schematic, debug=debug)
        # Reverse transform points to apply correct perspective transform to original image
        if reverse:
            src_pts_t = self.apply_reverse_affine(src_pts)
            dst_pts_t = reverse_image_transform(points=dst_pts, img_shape=image.shape, ird_match=match)
            if debug:
                plot_point_correspondences(image, self.ref_image, src_pts_t, dst_pts_t)
        else:
            src_pts_t = src_pts
            dst_pts_t = dst_pts
            if debug:
                plot_point_correspondences(affine_transformed, aligned_schematic, src_pts_t, dst_pts_t)

        # Calculate Homography between points
        H, _ = cv2.findHomography(
            srcPoints=src_pts_t[:, [1, 0]], 
            dstPoints=dst_pts_t[:, [1, 0]],
            method=cv2.RANSAC
        )
        return H, affine_transformed

    @timeit
    def undistort(
        self, 
        image: np.ndarray,
        chip_region: int,
        reverse: bool = True,
        debug: bool = False
    ) -> np.ndarray: 
        assert image.shape == self.setup_image.shape, f"Undistorting an image of different resolution than setup image. Setup image resolution: {self.setup_image.shape}, expected {self.setup_image.shape}, got {image.shape}"
        self.image = image
        self.reverse = reverse
        self.debug = debug

        if self.chip_region is None or chip_region != self.chip_region:
            self._initialize(image=image, chip_region=chip_region, debug=debug)

        if self.calculate_individual:
            H, affine_transformed = self.calculate_homography(
                image=image,
                reverse=reverse,
                debug=debug
            )
            
        else:
            H, affine_transformed = self.homography, None

        output_shape = (image.shape[1], image.shape[0])
        corrected_img = cv2.warpPerspective(image, H, output_shape)
    
        return corrected_img, affine_transformed, H


def combine_affine_matrices(T1: np.ndarray, T2: np.ndarray):
    # T1, T2: 2x3 affine matrices
    A1, b1 = T1[:, :2], T1[:, 2]
    A2, b2 = T2[:, :2], T2[:, 2]
    A = A1 @ A2
    b = A2 @ b1 + b2
    T_combined = np.zeros((2, 3))
    T_combined[:, :2] = A
    T_combined[:, 2] = b
    return T_combined


def extract_schematic(
    image: np.ndarray, 
    cropx: tuple = None, 
    threshold: float = 0.8*255, 
    min_size: int = 20,
    debug: bool = False
) -> np.ndarray:
    if cropx:
        image = image[:, cropx[0]:cropx[1]]
            
    # Thresholding
    binary = image < threshold  # This is a boolean mask

    # Area Filter individual components
    labeled = measure.label(binary)  # Now labeled has values 0, 1, 2, ...
    if labeled.max() > 0:
        filtered = morphology.remove_small_objects(labeled, min_size=min_size)
        mask = filtered > 0
    else:
        raise RuntimeError("No masks found.")

    if debug:
        # Plot results
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Original")
        
        axes[1].imshow(binary, cmap="gray")
        axes[1].set_title("Thresholded")
        
        axes[2].imshow(mask, cmap="gray")
        axes[2].set_title("Area Filtered")
        
        plt.tight_layout()
        plt.show()

    return mask


def bounding_box_from_masks(mc: "MaskContainer", debug: bool = False) -> [float, float]:
    """
    Given a list of binary masks, calculates the bounding box
    that encloses all foreground regions, and returns (width, height).
    """
    if len(mc) == 0:
        raise ValueError("Mask list is empty.")

    masks = [mask.segmentation for mask in mc.masks]

    # Combine all masks into one union mask
    union_mask = np.any(np.stack(masks), axis=0)

    # Label connected regions in the union
    labeled = label(union_mask)
    regions = regionprops(labeled)

    if not regions:
        raise ValueError("No foreground objects found in the masks.")

    # Compute the global bounding box over all regions
    minr = min(region.bbox[0] for region in regions)
    minc = min(region.bbox[1] for region in regions)
    maxr = max(region.bbox[2] for region in regions)
    maxc = max(region.bbox[3] for region in regions)

    width = maxc - minc
    height = maxr - minr

    if debug:
        # Plot the union mask with bounding box
        fig, ax = plt.subplots()
        ax.imshow(union_mask, cmap='gray')
        rect = plt.Rectangle((minc, minr), width, height,
                             edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)
        ax.set_title("Bounding Box Around All Masks")
        plt.axis('off')
        plt.show()

    return width, height


def pad_to_shape(img: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Center-pad `img` to match `target_shape`."""
    pad_height = target_shape[0] - img.shape[0]
    pad_width = target_shape[1] - img.shape[1]

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

def resize_uniform(image: np.ndarray, target_shape: tuple) -> [np.ndarray, float]:
    """Resize image uniformly to fit within target_shape (height, width)."""
    original_shape = np.array(image.shape[:2])  # (H, W)
    target_shape = np.array(target_shape)
    
    scale = np.min(target_shape / original_shape)
    new_shape = (original_shape * scale).astype(int)
    
    resized = resize(image, new_shape, preserve_range=True, anti_aliasing=False)
    return resized, scale


def calculate_resize(mc: "MaskContainer", debug: bool = False) -> tuple:
    w, h = bounding_box_from_masks(mc, debug=debug)
    return h, w


@timeit
def evaluate_camera_calibration(
    image: np.ndarray, 
    ref_image: np.ndarray, 
    segmenter: "Segmenter",
    use_binary: bool = True,
    debug: bool = False
) -> tuple[float, float, float, float]:
    """Undistorts the given image, template matches the schematic onto it using imreg_dft while maintaining aspect ratio, calculates overlap and resulting IoU."""

    segmenter.generate_masks(image)
    segmenter.filter_masks()

    # Perform similarity matching - use binary when matching to high RI image
    if use_binary:
        binary = segmenter.masks.as_binary()
        match = ird.similarity(binary, ref_image, numiter=3)
    else:
        match = ird.similarity(image, ref_image, numiter=3)

    if debug:
        print(f"Translation: {match['tvec']}")
        print(f"Scale: {match['scale']}")
        print(f"Rotation: {match['angle']}°")

    # Transform schematic to align with the image
    aligned_ref = ird.transform_img(
        img=ref_image, 
        scale=match['scale'], 
        angle=match['angle'], 
        tvec=match['tvec']
    )
    if debug:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        axes[0].imshow(image, cmap='gray')
        axes[0].set_title("Image")
        
        axes[1].imshow(image, cmap='gray')
        overlay = np.zeros((*aligned_ref.shape, 4))
        overlay[..., 2] = 1.0
        overlay[..., 3] = aligned_ref * 0.5
        
        axes[1].imshow(overlay)
        axes[1].set_title("Overlay: Image vs. Aligned Reference")
        
        for ax in axes:
            ax.axis("off")
        
        plt.tight_layout()
        plt.show()

    # Calculate IoU
    labeled = label(aligned_ref)
    gt_masks = [WellMask(segmentation=(labeled == i)) for i in range(1, labeled.max() + 1)]
    gt_mc = MaskContainer(image=aligned_ref, masks=gt_masks)

    iou_all = None
    iou_large = None
    iou_medium = None
    iou_small = None

    # Only use the masks that overlap for comparison
    mc, gt_mc = filter_overlapping_masks(segmenter.masks, gt_mc)
    iou_all = calculate_average_iou(gt_mc, mc)
    
    if segmenter.large_masks is not None and len(segmenter.medium_masks) >= 1:
        large_mc, large_gt_mc = filter_overlapping_masks(segmenter.large_masks, gt_mc)
        iou_large = calculate_average_iou(large_gt_mc, large_mc)

    if segmenter.medium_masks is not None and len(segmenter.medium_masks) >= 1:
        medium_mc, medium_gt_mc = filter_overlapping_masks(segmenter.medium_masks, gt_mc)
        iou_medium = calculate_average_iou(medium_gt_mc, medium_mc)
    
    if segmenter.small_masks is not None and len(segmenter.small_masks) >= 1:
        small_mc, small_gt_mc = filter_overlapping_masks(segmenter.small_masks, gt_mc)
        iou_small = calculate_average_iou(small_gt_mc, small_mc)

    if debug:
        print(f"Average IoU all: {iou_all or 0}")
        print(f"Average IoU large: {iou_large or 0}")
        print(f"Average IoU medium: {iou_medium or 0}")
        print(f"Average IoU small: {iou_small or 0}")
        
    return iou_all or 0.0, iou_large or 0.0, iou_medium or 0.0, iou_small or 0.0


def filter_overlapping_masks(mc: MaskContainer, gt_mc: MaskContainer, min_overlap_pixels: int = 5, max_center_dist: int = 50) -> [MaskContainer, MaskContainer]:
    overlapping = []
    overlapping_gt = []

    src_centers = np.array([m.center for m in mc.masks])
    gt_centers = np.array([m.center for m in gt_mc.masks])
    gt_tree = cKDTree(gt_centers)

    for mask in mc.masks:
        candidate_idx = gt_tree.query_ball_point(mask.center, r=max_center_dist)
        for j in candidate_idx:
            gt_mask = gt_mc.masks[j]
            overlap_pixels = np.sum(mask.segmentation & gt_mask.segmentation)
            if overlap_pixels > min_overlap_pixels:
                overlapping.append(mask)
                overlapping_gt.append(gt_mask)
                break
    return MaskContainer(image=mc.image, masks=overlapping), MaskContainer(image=gt_mc.image, masks=overlapping_gt)


def split_into_individual_masks(binary_image):
    # Ensure binary format (0 and 255), and uint8 type
    if binary_image.dtype == bool:
        binary_image = binary_image.astype(np.uint8) * 255
    elif binary_image.dtype != np.uint8:
        binary_image = (binary_image > 0).astype(np.uint8) * 255

    # Label connected components
    num_labels, labels = cv2.connectedComponents(binary_image, connectivity=8)

    # List to hold individual binary masks
    individual_masks = []

    for label in range(1, num_labels):  # skip background (label 0)
        mask = (labels == label).astype(np.uint8) * 255
        individual_masks.append(mask)

    return individual_masks


def plot_point_correspondences(img1, img2, pts1, pts2, point_labels=None, radius=5):
    """
    Plot two images side by side with corresponding points labeled.

    :param img1: Left image (NumPy array)
    :param img2: Right image (NumPy array)
    :param pts1: List or array of (x, y) points in img1
    :param pts2: List or array of (x, y) points in img2
    :param point_labels: Optional list of labels (e.g. ['P1', 'P2', ...])
    :param radius: Marker size for points
    """
    # Flip points from (y, x) to (x, y), since cv2 and matplotlib use different conventions
    pts1 = np.array(pts1)[:, [1, 0]]
    pts2 = np.array(pts2)[:, [1, 0]]

    # Ensure both images are 3-channel RGB
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    if len(img2.shape) == 2:
        img2 = (img2.astype(np.uint8)) * 255
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    
    # Combine images side by side
    h = max(img1.shape[0], img2.shape[0])
    w1, w2 = img1.shape[1], img2.shape[1]
    canvas = np.ones((h, w1 + w2, 3), dtype=np.uint8) * 255
    canvas[:img1.shape[0], :w1] = img1
    canvas[:img2.shape[0], w1:w1 + w2] = img2

    # Create plot
    plt.figure(figsize=(14, 6))
    plt.imshow(canvas.astype(np.uint8))
    plt.axis('off')

    # Draw and label each point
    for i, (p1, p2) in enumerate(zip(pts1, pts2)):
        label = point_labels[i] if point_labels else f'P{i+1}'
        color = np.random.rand(3,)  # random color for each point

        # Plot point in image 1
        plt.plot(p1[0], p1[1], 'o', color=color, markersize=radius)
        plt.text(p1[0]+5, p1[1]-5, label, color='gray', fontsize=10, fontweight='bold')

        # Plot point in image 2 (shifted in x)
        x2_shifted = p2[0] + w1
        plt.plot(x2_shifted, p2[1], 'o', color=color, markersize=radius)
        plt.text(x2_shifted+5, p2[1]-5, label, color='gray', fontsize=10, fontweight='bold')

    plt.title('Point Correspondences Between Images')
    plt.tight_layout()
    plt.show()

    # Convert back
    pts1 = np.array(pts1)[:, [1, 0]]
    pts2 = np.array(pts2)[:, [1, 0]]


def reverse_image_transform(points, img_shape, ird_match):
    """
    Reverse the transformations (embed, scale, rotate, shift) applied to an image for a set of points.
    
    Parameters:
        points: (N, 2) array of [x, y] points in final image
        img_shape: shape of the original image (height, width)
        scale: scale factor used in the transformation
        angle: rotation angle in degrees
        tvec: translation vector (dy, dx)
    
    Returns:
        reversed_points: (N, 2) array of [x, y] coordinates before transformation
    """
    tvec = ird_match['tvec']  # [ty, tx]
    angle = ird_match['angle']  # degrees
    scale = ird_match['scale']  # scaling factor

    points = np.asarray(points, dtype=float)
    img_shape = np.array(img_shape)
    
    # Match the padded canvas size used during phase-correlation registration (step 2)
    bigshape = np.round(img_shape * 1.2).astype(int)

    # Helper: compute how much the original image was padded (embed offset)
    def get_embed_offset(inner_shape, outer_shape):
        return ((outer_shape[1] - inner_shape[1]) / 2,  # x offset
                (outer_shape[0] - inner_shape[0]) / 2)  # y offset

    # Step 1: Reverse second embed_to (from bigshape back to transformed space)
    offset2 = get_embed_offset(img_shape, bigshape)
    points += offset2

    # Step 2: Reverse shift
    points -= np.array(tvec)

    # Step 3: Reverse rotation around center of bigshape
    theta = -np.deg2rad(angle)
    center_big = np.array(bigshape[::-1]) / 2  # shape (height, width) → (x, y)
    points -= center_big
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    points = points @ rot.T
    points += center_big

    # Step 4: Reverse scale (around center)
    points = (points - center_big) / scale + center_big

    # Step 5: Reverse initial embed_to (back to img coordinates)
    offset1 = get_embed_offset(img_shape, bigshape)
    points -= offset1

    return points


def find_overlapping_pairs(mc, gt_mc, min_overlap_pixels=5, max_center_dist=50):
    src_centers = np.array([m.center for m in mc.masks])
    gt_centers = np.array([m.center for m in gt_mc.masks])
    gt_tree = cKDTree(gt_centers)

    src_pts = []
    dst_pts = []

    for i, mask in enumerate(mc.masks):
        candidate_idx = gt_tree.query_ball_point(mask.center, r=max_center_dist)
        for j in candidate_idx:
            gt_mask = gt_mc.masks[j]
            overlap_pixels = np.sum(mask.segmentation & gt_mask.segmentation)
            if overlap_pixels > min_overlap_pixels:
                src_pts.append(mask.center)
                dst_pts.append(gt_mask.center)
                break
    return np.array(src_pts), np.array(dst_pts)