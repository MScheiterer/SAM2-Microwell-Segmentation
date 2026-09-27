import numpy as np


def intensity_analysis(gray_image, mask_container) -> dict:
    intensities = calculate_intensity(gray_image, mask_container)
    mean_intensity = np.mean(intensities)
    std_intensity = np.std(intensities)

    return {
        "masks": mask_container,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity
    }


def calculate_intensity(gray_image, mask_container, shrink_factor: float = 0.5):
    intensities = [np.mean(gray_image[mask.segmentation.astype(bool)]) for mask in mask_container.masks]
    valid_indices = two_sigma_filter(intensities) # Remove outliers using a 2σ filter
    
    intensities = [intensities[i] for i in valid_indices]
    mask_container.masks = [mask_container.masks[i] for i in valid_indices]

    return intensities


def two_sigma_filter(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    lower_thresh = mean_val - 2 * std_val
    upper_thresh = mean_val + 2 * std_val
    valid_indices = [i for i, item in enumerate(data)
                        if lower_thresh <= item <= upper_thresh]
    return valid_indices








    