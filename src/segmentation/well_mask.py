import numpy as np
from typing import Optional, Tuple, Union
import cv2
from matplotlib.patches import Circle, Ellipse, Polygon

def create_from_sam2_mask(image, sam2_mask):
    return WellMask(
        image=image,
        segmentation=sam2_mask["segmentation"],
        area=sam2_mask["area"],
        bbox=sam2_mask["bbox"],
        predicted_iou=sam2_mask["predicted_iou"]
    )
        

class WellMask:
    def __init__(
        self,
        segmentation: np.ndarray,
        mask_id: Optional[int] = None,
        image: Optional[np.ndarray] = None,
        image_path: Optional[str] = None,
        predicted_iou: Optional[float] = None,
        area: Optional[float] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        metadata: Optional[dict] = None
    ):
        self.original_segmentation = segmentation.astype(bool)
        self.image = image
        self.image_path = image_path
        self.predicted_iou = predicted_iou
        self.metadata = metadata or {}
        self.mask_id = mask_id

        self.approx_segmentation = None  # to be set later
        self.use_approx = False  # switch flag

        self._area = area
        self._bbox = bbox
        self._center = None
        self._contours = None

    @property
    def segmentation(self) -> np.ndarray:
        """Returns the current active mask."""
        return self.approx_segmentation if self.use_approx and self.approx_segmentation is not None else self.original_segmentation

    @property
    def area(self) -> float:
        if self._area is None:
            self._area = float(np.sum(self.segmentation > 0))
        return self._area

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        if self._bbox is None:
            y, x = np.nonzero(self.segmentation)
            self._bbox = (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)
        return self._bbox

    @property
    def center(self) -> Tuple[int, int]:
        if self._center is None:
            y, x = np.nonzero(self.segmentation)
            if len(y) == 0 or len(x) == 0:
                self._center = (0, 0)
                print("==== Warning: mask is probably empty ====")
            else:
                self._center = (int(np.round(y.mean())), int(np.round(x.mean())))
        return self._center

    @property
    def contours(self):
        if self._contours is None:
            contours, _ = cv2.findContours(self.segmentation.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            self._contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours] # Try to smooth contours
        return self._contours

    def distance_to(self, other: "WellMask") -> float:
        y1, x1 = self.center
        y2, x2 = other.center
        return float(np.hypot(y2 - y1, x2 - x1))

    def __repr__(self):
        return (f"WellMask(mask_id={self.mask_id}, area={self.area}, center={self.center})")

    def calculate_approximation(self, method: str, shrink_factor: float = 0.7):
        """
        Approximate the mask of a WellMask using a circle or ellipse.
        Stores the result in the .approx_segmentation field.
    
        Parameters:
            well_mask: WellMask object with .original_segmentation
            method: 'circle' or 'ellipse'
            shrink_factor: multiplier to reduce radius/axes
    
        Returns:
            True if approximation was successful, False otherwise
        """
        if self.metadata.get("method") == method:
            return # check if approx has already been calculated

        mask = self.segmentation.astype(np.uint8)
        contours = self.contours

        contour = max(contours, key=cv2.contourArea)
    
        if method == "circle":
            (x, y), radius = cv2.minEnclosingCircle(contour)
        
            # Shrink to avoid edges
            adjusted_radius = int(radius * shrink_factor)
            center = (int(round(y)), int(round(x)))  # return in (y, x) format
        
            circle_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
            cv2.circle(circle_mask, (center[1], center[0]), adjusted_radius, 255, -1)

            self.set_approximation(
                circle_mask, method=method, shrink_factor=shrink_factor,
                center=center, radius=adjusted_radius
            )
    
        elif method == "ellipse":
            if len(contour) < 5:
                return False  # not enough points for ellipse fit
    
            (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(contour)
            center = (int(round(y)), int(round(x)))  # (y, x)
        
            ellipse_mask = np.zeros(mask.shape, dtype=np.uint8)
        
            scaled_major = shrink_factor * major_axis
            scaled_minor = shrink_factor * minor_axis
            scaled_axes = (scaled_major, scaled_minor)
            cv2.ellipse(ellipse_mask, ((x, y), scaled_axes, angle), 255, thickness=-1)
            
            self.set_approximation(
                ellipse_mask, method=method, shrink_factor=shrink_factor,
                center=center, axes=scaled_axes, angle=angle
            )
    
        else:
            raise ValueError("Method must be 'circle' or 'ellipse'")
        return True

    def set_approximation(
        self,
        approx_mask: np.ndarray,
        method: str,
        shrink_factor: float,
        center: Optional[Tuple[int, int]] = None,
        radius: Optional[float] = None,
        axes: Optional[Tuple[float, float]] = None,
        angle: Optional[float] = None
    ):
        self.approx_segmentation = approx_mask.astype(np.uint8)
        self.metadata['approximation'] = {
            'method': method,
            'shrink_factor': shrink_factor
        }
        self.approximation_info = {
            "center": center,
            "radius": radius,
            "axes": axes,
            "angle": angle
        }

    def use_approximated(self, flag: bool = True, method: str = "circle", shrink_factor: float = 0.7):
        """Toggle between original and approximated mask."""
        self.use_approx = flag
        if flag:
            self.calculate_approximation(method=method, shrink_factor=shrink_factor)

    def visualize_mask(self, ax, img, color, identify: bool = False):
        label_text = str(self.mask_id) if hasattr(self, "mask_id") else ""
        
        if self.use_approx and self.metadata["approximation"]["method"] == "circle":
            center = self.approximation_info["center"]
            radius = self.approximation_info["radius"]
            circle = Circle((center[1], center[0]), radius, edgecolor=color[:3], facecolor=color, linewidth=1)
            ax.add_patch(circle)

            if identify:
                ax.text(center[1], center[0], label_text, color='black', ha='center', va='center', fontsize=8, weight='bold')
            
        elif self.use_approx and self.metadata["approximation"]["method"] == "ellipse":
            cy, cx = self.approximation_info["center"]
            major, minor = self.approximation_info["axes"]
            angle = self.approximation_info["angle"]
            ellipse = Ellipse((cx, cy), major, minor, angle=angle, edgecolor=color[:3], facecolor=color, linewidth=1)
            ax.add_patch(ellipse)

            if identify:
                ax.text(cx, cy, label_text, color='black', ha='center', va='center', fontsize=8, weight='bold')
            
        else:
            img[self.segmentation] = color
            for contour in self.contours:
                contour = contour.squeeze()
                if contour.ndim == 2:  # avoid errors with single points
                    polygon = Polygon(contour, closed=True, edgecolor=color[:3], facecolor='none', linewidth=1)
                    ax.add_patch(polygon)

            cy, cx = self.center  # Use original mask center

            if identify:
                ax.text(cx, cy, label_text, color='black', ha='center', va='center', fontsize=8, weight='bold')


