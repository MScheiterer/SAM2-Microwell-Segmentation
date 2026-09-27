"""
Provides the following noise correction algorithms:
- Flat-field Correction: written as a class to be easily extendable, e.g. to save flat-field image or coefficient, camera settings, darkfield image, etc.
- 
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageEnhance

class FlatFieldCorrector:
    def __init__(self):
        pass

    def approximate_darkfield(self, gray_image: np.ndarray, sigma: int = 5) -> np.ndarray:
        # Estimate Darkfield by dimming image
        pil_img = Image.fromarray(gray_image)
        
        darkfield = ImageEnhance.Brightness(pil_img)
        darkfield = darkfield.enhance(0)  # Reduce brightness to simulate a dark field (reduced by 100%)
        dark_array = np.array(darkfield)
        
        return dark_array
    
    def approximate_flatfield(self, gray_image: np.ndarray, sigma: int = 100) -> np.ndarray:
        flatfield = gaussian_filter(gray_image.astype(np.float64), sigma=sigma)
        return flatfield
    
    def apply_flatfield_correction(self, gray_image: np.ndarray, flatfield: np.ndarray = None, darkfield: np.ndarray = None) -> np.ndarray:
        if flatfield is None:
            flatfield = self.approximate_flatfield(gray_image)

        if darkfield is None:
            darkfield = self.approximate_darkfield(gray_image)
        
        # Change to float64 for accurate division
        image_float = gray_image.astype(np.float64)
        flatfield_float = flatfield.astype(np.float64)
        darkfield_float = darkfield.astype(np.float64)
        
        denominator = flatfield_float - darkfield_float
        epsilon = 1e-6
        denominator[denominator == 0] = epsilon  # Avoid division by zero

        mean_intensity = np.mean(flatfield_float - darkfield_float)
        corrected = (image_float - darkfield_float) / denominator * mean_intensity 
        
        # Clip values to valid range
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        return corrected, flatfield, darkfield

    def visualize_flatfield_correction(self, original: np.ndarray, flatfield: np.ndarray, darkfield: np.ndarray, corrected: np.ndarray):
        """
        Plot original, flatfield, and corrected images side-by-side.
        """
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 4, 1); plt.imshow(original, cmap='gray'); plt.title("Original"); plt.axis('off')
        plt.subplot(1, 4, 2); plt.imshow(flatfield, cmap='gray'); plt.title("Estimated Flatfield"); plt.axis('off')
        plt.subplot(1, 4, 3); plt.imshow(darkfield, cmap='gray'); plt.title("Estimated Darkfield"); plt.axis('off')
        plt.subplot(1, 4, 4); plt.imshow(corrected, cmap='gray'); plt.title("Corrected"); plt.axis('off')
        plt.tight_layout()
        plt.show()



        