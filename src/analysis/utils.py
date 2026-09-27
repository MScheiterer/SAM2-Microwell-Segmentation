import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.stats import linregress
import pandas as pd
import cv2
from segmentation.mask_container import MaskContainer
import os
import time
import functools
import os
from pathlib import Path
from analysis.noise_correction import FlatFieldCorrector


def apply_preprocessing(
    image: np.ndarray, 
    calibrator: "CameraCalibrator",
    cc: bool, 
    ffc: bool, 
    chip_region: int,
    debug: bool = False
) -> np.ndarray:
    result = image
    corrector = FlatFieldCorrector()

    if cc and ffc:
        corrected, _, _ = corrector.apply_flatfield_correction(image)
        calibrated, _, _ = calibrator.undistort(corrected, chip_region=chip_region, debug=debug)
        result = calibrated
    else:
        if cc:
            calibrated, _, _ = calibrator.undistort(image, chip_region=chip_region, debug=debug)
            result = calibrated
        elif ffc:
            corrected, flatfield, darkfield = corrector.apply_flatfield_correction(image)
            result = corrected
    
    # Visualize Results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(result, cmap='gray')
    plt.title(f'Preprocessed Image: cc: {cc}, ffc: {ffc}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    return result


def create_mask_plots(calibration_data, identify: bool = True):
    """
    Visualizes masks for all images in a batch of calibration_data. 
    """
    n = len(calibration_data)
    
    cols = int(n**0.5)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * rows, 4 * cols))
    axes = axes.flatten() if n > 1 else [axes]  # Ensure iterable

    for i, elem in enumerate(sorted(calibration_data, key=lambda x: x["NaCl_percentage"])):
        masks_large = elem["masks_large"]
        masks_medium = elem["masks_medium"]
        masks_small = elem["masks_small"]
        img = elem["image"]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axes[i].imshow(img)
        masks_large.plot_all_wells(ax=axes[i], color=(200/255, 160/255, 255/255, 0.4), identify=identify) # light purple
        masks_medium.plot_all_wells(ax=axes[i], color=(0/255, 255/255, 255/255, 0.4), identify=identify) # cyan
        masks_small.plot_all_wells(ax=axes[i], color=(180/255, 140/255, 0/255, 0.4), identify=identify)  # dark yellow
        axes[i].set_title(f"{elem['NaCl_percentage']}%")
        axes[i].axis('off')
        
    # Hide unused subplots, if any
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def create_intensity_plots(df_calibration: pd.DataFrame, well_sizes: list, mask_containers: dict, individual_intensities: bool = False, csv_path: str = None, settings: str = None, model: str = None):
    n = len(well_sizes)
    
    cols = int(n**0.5)
    rows = (n + cols - 1) // cols

    if individual_intensities:
        figsize = (15,45)
    else:
        figsize = (5,15)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]  # Ensure iterable

    slopes = {}
    
    for i, well_size in enumerate(well_sizes):
        slope = plot_intensity_vs_nacl(
            df=df_calibration, 
            group=well_size, 
            ax=axes[i], 
            use_approx=True, 
            individual_intensities=individual_intensities,
            csv_path=csv_path,
            settings=settings,
            model=model
        )
        axes[i].grid(False)
        slopes[well_size] = slope

    # Hide unused subplots, if any
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
    return slopes
    

def plot_intensity_vs_nacl(df: pd.DataFrame, group: str = "medium", ax: plt.axes = None, 
                          figsize: tuple = (10, 6), title: str = None, use_approx: bool = False, 
                          individual_intensities: bool = False, csv_path: str = None, 
                          settings: str = "default", model: str = "SAM2"):
    """
    Plots average and individual well intensities vs NaCl concentration for a given group.
    Saves analysis results to CSV file.
    
    Parameters:
    - df: DataFrame containing 'NaCl_percentage' and intensity/mask data
    - group: one of 'small', 'medium', 'large', or 'all'
    - ax: optional matplotlib axis
    - figsize: plot size if ax is None
    - title: optional title
    - use_approx: use approximated masks if available
    - individual_intensities: whether to plot individual well intensities
    - csv_path: path to CSV file for saving results (if None, no saving occurs)
    - settings: settings parameter to save in CSV
    - model: model parameter to save in CSV
    """
    if group not in ['large', 'medium', 'small', 'all']:
        raise ValueError("Group must be one of: 'large', 'medium', 'small', 'all'")
    
    x = df["RI"].values
    y = df[f"mean_intensity_{group}"].values
    yerr = df[f"std_intensity_{group}"].values
    
    # Linear regression on mean intensities
    slope, intercept, r, _, _ = linregress(x, y)
    r2 = r**2
    std_dev = np.std(yerr)  # Standard deviation of the error bars
    
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept
    
    # Save results to CSV if path is provided
    if csv_path is not None:
        nacl_percentages = df['NaCl_percentage'].values
        save_results_to_csv(csv_path, settings, model, group, slope, intercept, r2, std_dev,
                           x, y, yerr, nacl_percentages)
    
    # Create axis if needed
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    
    # Plot mean intensities with error bars
    ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=5,
                label=f'{group.capitalize()} Mean', color='tab:blue')
    
    # Annotate points with NaCl percentages
    for i, row in df.iterrows():
        ax.annotate(f"{row['NaCl_percentage']}%",
                     (row['RI'], row[f'mean_intensity_{group}']),
                     xytext=(5, 5), textcoords='offset points')
    
    # Plot individual well intensities
    if individual_intensities:
        for row in df.itertuples():
            ri = row.RI
            mask_container = getattr(row, f"masks_{group}", [])
    
            for well in mask_container.masks:
                if use_approx and well.approx_segmentation is not None:
                    mask = well.approx_segmentation > 0
                else:
                    mask = well.segmentation > 0
    
                intensity = float(np.mean(row.image[mask]))
                mask_id = getattr(well, 'mask_id', '?')
                
                ax.scatter(ri, intensity, color='tab:green', s=20, alpha=0.7)
                ax.annotate(str(mask_id),
                         (ri, intensity),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot regression line
    fit_label = f"Fit: y = {slope:.4f}x + {intercept:.4f} \n(R² = {r2:.4f})"
    ax.plot(x_fit, y_fit, linestyle='--', color='tab:red', label=fit_label)
    
    # Final touches
    ax.set_xlabel("Refractive Index")
    ax.set_ylabel("Mean Intensity (AU)")
    ax.set_title(title or f"Intensity vs RI – {group.capitalize()} Wells")
    ax.grid(True)
    ax.legend(loc="upper left")
    
    if created_fig:
        plt.tight_layout()
        plt.show()
        
    return slope


def save_results_to_csv(csv_path: str, settings: str, model: str, group: str, 
                       slope: float, intercept: float, r2: float, std_dev: float,
                       ri_values: np.ndarray, mean_intensities: np.ndarray, 
                       error_bars: np.ndarray, nacl_percentages: np.ndarray):
    """
    Save or append analysis results to a CSV file with complete plot recreation data.
    
    Parameters:
    - csv_path: path to the CSV file
    - settings: settings parameter
    - model: model parameter  
    - group: well group (small, medium, large, all)
    - slope: regression slope
    - intercept: y-intercept of regression line
    - r2: R-squared value
    - std_dev: standard deviation of error bars
    - ri_values: array of refractive index values
    - mean_intensities: array of mean intensity values
    - error_bars: array of error bar values (std deviations)
    - nacl_percentages: array of NaCl percentage values
    """
    # Create directory if it doesn't exist
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert arrays to semicolon-separated strings for storage
    ri_str = ';'.join([f"{val:.6f}" for val in ri_values])
    intensities_str = ';'.join([f"{val:.6f}" for val in mean_intensities])
    errors_str = ';'.join([f"{val:.6f}" for val in error_bars])
    nacl_str = ';'.join([f"{val:.2f}" for val in nacl_percentages])
    
    # Prepare new row data
    new_row = {
        'settings': settings,
        'model': model,
        'group': group,
        'slope': slope,
        'intercept': intercept,
        'r2': r2,
        'std_deviation': std_dev,
        'ri_values': ri_str,
        'mean_intensities': intensities_str,
        'error_bars': errors_str,
        'nacl_percentages': nacl_str,
        'n_points': len(ri_values)
    }
    
    # Check if file exists
    if os.path.exists(csv_path):
        # Read existing data
        try:
            existing_df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            # File exists but is empty
            existing_df = pd.DataFrame()
        
        # Append new row
        if not existing_df.empty:
            updated_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            updated_df = pd.DataFrame([new_row])
    else:
        # Create new DataFrame
        updated_df = pd.DataFrame([new_row])
    
    # Save to CSV
    updated_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[{func.__name__}] executed in {end - start:.4f} seconds")
        return result
    return wrapper    
