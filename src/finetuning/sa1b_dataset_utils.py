import os
import json
import shutil
import numpy as np
from pycocotools import mask as mask_utils
import cv2
import matplotlib.pyplot as plt


def construct_masks_from_json(path):
    """
    Construct segmentation masks from Label Studio JSON annotations containing ellipse annotations.
    
    This function processes Label Studio export files that contain ellipse annotations and converts
    them into COCO-format segmentation masks with RLE (Run-Length Encoding) compression.
    
    Args:
        path (str): Path to the JSON file containing Label Studio annotations.
                   Expected format: Label Studio json export with ellipse annotations.
    
    Returns:
        list: A list of dictionaries, each containing:
            - image (dict): Image metadata with keys:
                - image_id (str): Image identifier (filename without extension)
                - width (int): Image width in pixels
                - height (int): Image height in pixels  
                - file_name (str): Original image filename
            - annotations (list): List of annotation dictionaries, each with:
                - bbox (list): Bounding box as [x, y, width, height]
                - area (int): Area of the segmentation mask in pixels
                - segmentation (dict): RLE-encoded mask with 'size' and 'counts'
                - type (str): Ellipse classification ("large", "medium", "small", or "well")
    """
    data = []
    
    # Load and parse the JSON annotation file
    with open(path, 'r') as f:
        json_data = json.load(f)
        
        # Process each annotated image in the dataset
        for json_item in json_data:
            # Extract image name by removing the first part before the first dash
            # This handles Label Studio's file naming convention
            img_name = "-".join(json_item["file_upload"].split("-")[1:])
            
            # Get all ellipse annotations for this image
            ellipses = json_item["annotations"][0]["result"]
            
            annotations = []
            
            # Process each ellipse annotation
            for ellipse in ellipses:
                # Extract image dimensions from the annotation
                height = ellipse["original_height"]
                width = ellipse["original_width"]
                
                # Extract ellipse type/classification (first word of the label)
                ellipse_type = ellipse["value"]["ellipselabels"][0].split(" ")[0]
                
                # Convert ellipse parameters to RLE-encoded segmentation mask
                area, bbox, segmentation = ellipse_to_rle_mask(
                    height, width, 
                    ellipse["value"], 
                    ellipse["image_rotation"]
                )
                
                # Create annotation entry in COCO format
                annotations.append({
                    "bbox": bbox,                    # [x, y, width, height]
                    "area": area,                    # Total pixels in the mask
                    "segmentation": {                # RLE-encoded segmentation
                        "size": [height, width],
                        "counts": segmentation["counts"]
                    },
                    "type": ellipse_type            # Classification: "large", "medium", "small", "well"
                })
            
            # Create image entry with metadata and all its annotations
            data.append({
                "image": {
                    "image_id": img_name.split(".")[0],  # Remove file extension
                    "width": width,
                    "height": height,
                    "file_name": img_name,
                },
                "annotations": annotations
            })
    
    return data


def ellipse_to_rle_mask(height, width, ellipse, image_rotation):
    """
    Convert ellipse parameters to RLE-encoded segmentation mask.
    
    This function takes ellipse parameters (center, radii, rotation) and generates
    a binary segmentation mask, then encodes it using COCO's RLE format for
    efficient storage and processing.
    
    Args:
        height (int): Image height in pixels
        width (int): Image width in pixels  
        ellipse (dict): Ellipse parameters containing:
            - x (float): Center x-coordinate as percentage (0-100)
            - y (float): Center y-coordinate as percentage (0-100)
            - radiusX (float): Horizontal radius as percentage (0-100)
            - radiusY (float): Vertical radius as percentage (0-100)
            - rotation (float): Ellipse rotation angle in degrees
        image_rotation (float): Additional image-level rotation in degrees
    
    Returns:
        tuple: A 3-tuple containing:
            - area (int): Number of pixels inside the ellipse
            - bbox (list): Bounding box as [x, y, width, height] in pixels
            - segmentation (dict): RLE-encoded mask with 'size' and 'counts' keys
    
    Note:
        The ellipse parameters are given as percentages and are converted to
        absolute pixel coordinates. The total rotation is the sum of the
        ellipse rotation and any image-level rotation.
    """
    # Generate the binary mask by drawing the ellipse
    mask, area, bbox = draw_mask(height, width, ellipse, image_rotation)
    
    # Convert mask values from 255 to 1 for binary representation
    binary_mask = np.where(mask > 0, 1, 0).astype(np.uint8)
    
    # Encode using COCO RLE format (requires Fortran-style memory order)
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    
    # Convert RLE 'counts' from bytes to UTF-8 string for JSON serialization
    rle["counts"] = rle["counts"].decode("utf-8")
    
    return area, bbox, {
        "size": [height, width],
        "counts": rle["counts"]
    }


def draw_mask(height, width, ellipse, image_rotation):
    """
    Draw an ellipse mask on a canvas and compute its properties.
    
    Creates a binary mask by drawing a filled ellipse using OpenCV, then
    calculates the area and bounding box of the resulting shape.
    
    Args:
        height (int): Canvas height in pixels
        width (int): Canvas width in pixels
        ellipse (dict): Ellipse parameters with percentage-based coordinates:
            - x (float): Center x-coordinate (0-100% of width)
            - y (float): Center y-coordinate (0-100% of height)  
            - radiusX (float): Horizontal radius (0-100% of width)
            - radiusY (float): Vertical radius (0-100% of height)
            - rotation (float): Ellipse rotation angle in degrees
        image_rotation (float): Additional rotation to apply in degrees
    
    Returns:
        tuple: A 3-tuple containing:
            - canvas (numpy.ndarray): Binary mask as uint8 array (0 or 255)
            - area (int): Number of non-zero pixels in the mask
            - bbox (list): Tight bounding box as [x, y, width, height]
    
    Note:
        - Coordinates are converted from percentages to absolute pixels
        - The ellipse is drawn filled (cv2.FILLED or -1 thickness)
        - Bounding box is the minimal rectangle containing all non-zero pixels
    """
    # Create blank canvas (all zeros)
    canvas = np.zeros((height, width), dtype=np.uint8)
    
    # Convert percentage-based coordinates to absolute pixel coordinates
    x = ellipse["x"] / 100 * width           # Center x in pixels
    y = ellipse["y"] / 100 * height          # Center y in pixels
    radius_x = ellipse["radiusX"] / 100 * width   # Horizontal radius in pixels
    radius_y = ellipse["radiusY"] / 100 * height  # Vertical radius in pixels
    rotation = ellipse["rotation"]            # Ellipse rotation angle
    
    # Calculate total rotation (ellipse + image rotation)
    total_rotation = rotation + image_rotation
    
    # Prepare parameters for OpenCV ellipse drawing
    center = (int(x), int(y))                # Center point as integer coordinates
    axes = (int(radius_x), int(radius_y))    # Radii as integer values
    
    # Draw filled ellipse on canvas
    # Parameters: image, center, axes, angle, start_angle, end_angle, color, thickness
    cv2.ellipse(canvas, center, axes, total_rotation, 0, 360, 255, -1)
    
    # Calculate area (number of non-zero pixels)
    area = int(np.count_nonzero(canvas))
    
    # Find bounding box of the drawn ellipse
    coords = cv2.findNonZero(canvas)         # Get all non-zero pixel coordinates
    x, y, w, h = cv2.boundingRect(coords)    # Compute minimal bounding rectangle
    bbox = [float(x), float(y), float(w), float(h)]  # Convert to float list
    
    return canvas, area, bbox


def remove_commas_percentages(path, name):
    """
    Find the actual filename by removing commas and percentages from the search name.
    
    This function addresses Label Studio export issues where image filenames may have
    commas and percentage signs removed during the annotation process. It searches
    for files that match the given name after removing these characters.
    
    Args:
        path (str): Directory path to search for image files
        name (str): Target filename to search for (may have missing commas/percentages)
    
    Returns:
        str or None: The actual filename if found, None otherwise
    
    Note:
        This is a workaround for Label Studio export inconsistencies where special
        characters in filenames are sometimes changed during the export process.
    """
    # Get all files in the specified directory
    files = os.listdir(path)
    
    # Search through all JPEG files
    for file in files:
        if file.endswith(".jpg"):
            # Create normalized version by removing commas and percentages
            new_name = file.replace(",", "")      # Remove commas
            new_name = new_name.replace("%", "")  # Remove percentage signs

            # Check if normalized name matches the search target
            if new_name == name:
                return file  # Return the original filename
    
    # Debug output if no match found
    print(f"New name: {new_name}, Looking for: {name}")
    return None  # Explicitly return None if no match found


def create_sa1b_dataset(data, input_im_dir, output_im_dir, output_ann_dir):
    """
    Create a SA-1B (Segment Anything 1 Billion) formatted dataset from processed annotation data.
    
    This function takes structured annotation data and creates a standardized dataset by:
    1. Renaming images with sequential numeric IDs (1.jpg, 2.jpg, etc.)
    2. Copying images to the output directory with new names
    3. Updating metadata to reflect the new naming scheme
    4. Saving individual JSON annotation files for each image
    
    The SA-1B format expects each image to have a corresponding JSON file with the same
    base name, containing all annotations for that image.
    
    Args:
        data (list): List of dictionaries containing image and annotation data.
                    Each dictionary should have the structure:
                    {
                        "image": {
                            "file_name": str,
                            "image_id": str/int,
                            "width": int,
                            "height": int
                        },
                        "annotations": [list of annotation dictionaries]
                    }
        input_im_dir (str): Path to directory containing the original images
        output_im_dir (str): Path to directory where renamed images will be saved
        output_ann_dir (str): Path to directory where JSON annotation files will be saved
    
    Note:
        - Images are renamed sequentially starting from 1.jpg
        - Original filenames are preserved in the annotation metadata
        - The function handles filename normalization issues from Label Studio exports
        - Each image gets its own JSON file with all its annotations
    """
    # Process each annotation entry in the dataset
    for idx, item in enumerate(data, start=1):
        # Generate new sequential filenames (1.jpg, 2.jpg, etc.)
        new_name = f"{idx}.jpg"
        new_json_name = f"{idx}.json"
        
        # Handle potential filename normalization issues from Label Studio exports
        # This accounts for cases where commas/percentages may have been removed
        old_image_name = remove_commas_percentages(input_im_dir, item["image"]["file_name"])
        
        # Construct full paths for source and destination
        src_path = os.path.join(input_im_dir, old_image_name)
        dst_path = os.path.join(output_im_dir, new_name)
        
        # Copy and rename the image file
        # Using shutil.copy preserves file permissions and timestamps
        try:
            shutil.copy(src_path, dst_path)
        except FileNotFoundError:
            print(f"Warning: Source image not found: {src_path}")
            continue
        except Exception as e:
            print(f"Error copying {src_path} to {dst_path}: {e}")
            continue
        
        # Update the image metadata to reflect the new naming scheme
        item["image"]["file_name"] = new_name    # Update filename
        item["image"]["image_id"] = idx          # Update ID to sequential number
        
        # Save the updated annotation data as individual JSON file
        json_path = os.path.join(output_ann_dir, new_json_name)
        try:
            with open(json_path, "w") as f:
                # Use indent=2 for readable JSON formatting
                json.dump(item, f, indent=2)
        except Exception as e:
            print(f"Error saving annotation file {json_path}: {e}")
            continue
    
    print(f"Successfully processed {len(data)} images and annotations")


def visualize_annotation(annotation_dict, image_dir):
    """
    Visualize segmentation annotations overlaid on the original image.
    
    This function creates a color-coded visualization of segmentation masks overlaid
    on the original image. Different annotation types are displayed in different colors
    to help distinguish between various classes of objects (e.g., large, medium, small, well).
    
    Args:
        annotation_dict (dict): Dictionary containing image metadata and annotations.
                               Expected structure:
                               {
                                   "image": {
                                       "file_name": str,
                                       "height": int,
                                       "width": int,
                                       "image_id": str/int
                                   },
                                   "annotations": [
                                       {
                                           "segmentation": dict,  # RLE-encoded mask
                                           "type": str,           # "large", "medium", "small", "well"
                                           "bbox": list,          # [x, y, width, height]
                                           "area": int            # Area in pixels
                                       }
                                   ]
                               }
        image_dir (str): Path to directory containing the image files
    
    Returns:
        None: Function displays the visualization using matplotlib and doesn't return a value
    
    Note:
        - Colors are assigned based on annotation type:
          * large: Red (255, 0, 0)
          * medium: Green (0, 255, 0)  
          * small: Blue (0, 0, 255)
          * well: Yellow (255, 255, 0)
        - The overlay uses 70% original image and 30% colored mask for visibility
        - Multiple overlapping masks are combined using maximum operation
    """
    # Extract image metadata
    image_filename = annotation_dict["image"]["file_name"]
    height = annotation_dict["image"]["height"]
    width = annotation_dict["image"]["width"]
    
    # Load the original image
    image_path = os.path.join(image_dir, image_filename)
    image = cv2.imread(image_path)
    
    # Error handling for missing images
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Convert from BGR (OpenCV default) to RGB (matplotlib default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a blank canvas for combining all masks
    # This will be used to track all annotated regions
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Define color mapping for different annotation types
    # Each type gets a distinct color for easy visual identification
    type_colors = {
        "large": [255, 0, 0],    # Red - typically for large objects/cells
        "medium": [0, 255, 0],   # Green - for medium-sized objects/cells
        "small": [0, 0, 255],    # Blue - for small objects/cells
        "well": [255, 255, 0]    # Yellow - for well boundaries or special regions
    }
    
    # Create overlay image (copy of original for color modifications)
    overlay = image.copy()
    
    # Process each annotation in the current image
    for ann in annotation_dict["annotations"]:
        # Extract RLE-encoded segmentation mask
        rle = ann["segmentation"]
        
        # Decode RLE mask to binary array (0s and 1s)
        # This converts the compressed mask back to a 2D binary array
        binary_mask = mask_utils.decode(rle)
        
        # Apply color overlay to regions where mask == 1
        # This colors all pixels within the annotated region
        annotation_type = ann["type"]
        if annotation_type in type_colors:
            overlay[binary_mask == 1] = type_colors[annotation_type]
        else:
            # Default color for unknown types (white)
            print(f"Warning: Unknown annotation type '{annotation_type}', using white color")
            overlay[binary_mask == 1] = [255, 255, 255]
        
        # Combine current mask with previous masks using element-wise maximum
        # This ensures overlapping regions are preserved
        combined_mask = np.maximum(combined_mask, binary_mask)
    
    # Create final visualization by blending original image with colored overlay
    # Weights: 70% original image (0.7) + 30% colored overlay (0.3)
    # This maintains image details while showing annotations clearly
    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    # Display the result using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(result)
    plt.title(f"Segmentation Overlay: {image_filename}")
    plt.axis("off")  # Hide axis labels and ticks for cleaner visualization
    plt.show()
