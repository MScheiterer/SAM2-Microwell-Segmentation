import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil
from collections import defaultdict


def create_dataset_split(
    chip2_path: str,
    chip4_path: str,
    chip2_preprocessed_path: str,
    chip4_preprocessed_path: str,
    output_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    copy_files: bool = True
) -> Dict[str, List[str]]:
    """
    Create train/validation/test split for SAM2 training without data leakage.
    
    This function creates a proper dataset split for training SAM2 (Segment Anything Model 2)
    by splitting multiple related datasets while ensuring no data leakage between splits.
    It handles both original and preprocessed versions of datasets, creating combined
    datasets suitable for SAM2 training with proper file organization and metadata.
    
    Args:
        chip2_path (str): Path to chip2 dataset containing Images/ and Annotations/ folders
        chip4_path (str): Path to chip4 dataset containing Images/ and Annotations/ folders
        chip2_preprocessed_path (str): Path to preprocessed chip2 dataset
        chip4_preprocessed_path (str): Path to preprocessed chip4 dataset
        output_path (str): Path where the split datasets will be created
        train_ratio (float, optional): Percentage of data for training (0.0-1.0). Defaults to 0.7.
        val_ratio (float, optional): Percentage of data for validation (0.0-1.0). Defaults to 0.15.
        test_ratio (float, optional): Percentage of data for testing (0.0-1.0). Defaults to 0.15.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        copy_files (bool, optional): Whether to copy files or just create file lists. Defaults to True.
        
    Returns:
        Dict[str, List[str]]: Dictionary containing split information with file lists for each split.
                             Structure: {
                                 'train': {'chip2': [...], 'chip4': [...]},
                                 'val': {'chip2': [...], 'chip4': [...]},
                                 'test': {'chip2': [...], 'chip4': [...]}
                             }
    
    Raises:
        ValueError: If train_ratio + val_ratio + test_ratio != 1.0
        ValueError: If no valid files found in any dataset
        FileNotFoundError: If required dataset paths don't exist
    
    Output Structure:
        output_path/
        ├── train/
        │   ├── chip2/{Images/, Annotations/}
        │   ├── chip4/{Images/, Annotations/}
        │   ├── chip2_preprocessed/{Images/, Annotations/}
        │   ├── chip4_preprocessed/{Images/, Annotations/}
        │   └── combined/{Images/, Annotations/}  # All datasets merged with prefixes
        ├── val/        # Same structure as train/
        ├── test/       # Same structure as train/
        ├── split_metadata.json
        └── *_files.txt  # File lists for SAM2 training
    
    Note:
        - Ensures no data leakage by splitting at the file ID level across all datasets
        - Creates both individual dataset splits and combined datasets for SAM2 training
        - Generates file lists compatible with SAM2 training scripts
        - Validates that corresponding files exist in both original and preprocessed datasets
        - Uses prefixes (chip2_, chip4_, etc.) to avoid naming conflicts in combined datasets
    """
    
    # Validate that split ratios sum to 1.0 (allow small floating point errors)
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"train_ratio + val_ratio + test_ratio must equal 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    # Set random seed for reproducible splits across runs
    random.seed(seed)
    
    # Create output directory structure
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define split names and dataset types
    splits = ['train', 'val', 'test']
    dataset_types = ['chip2', 'chip4', 'chip2_preprocessed', 'chip4_preprocessed']
    
    # Create all necessary subdirectories
    for split in splits:
        for dataset_type in dataset_types:
            (output_path / split / dataset_type / 'Images').mkdir(parents=True, exist_ok=True)
            (output_path / split / dataset_type / 'Annotations').mkdir(parents=True, exist_ok=True)
    
    def get_available_files(dataset_path: str) -> List[str]:
        """
        Get list of available file IDs from a dataset by checking for matching image/annotation pairs.
        
        Args:
            dataset_path (str): Path to dataset containing Images/ and Annotations/ folders
            
        Returns:
            List[str]: Sorted list of file IDs (without extensions) that have both image and annotation
        """
        images_path = Path(dataset_path) / 'Images'
        if not images_path.exists():
            print(f"Warning: {images_path} does not exist")
            return []
        
        file_ids = []
        # Find all JPG images and check for corresponding annotations
        for img_file in images_path.glob('*.jpg'):
            file_id = img_file.stem  # Get filename without extension
            
            # Check if corresponding JSON annotation exists
            ann_file = Path(dataset_path) / 'Annotations' / f'{file_id}.json'
            if ann_file.exists():
                file_ids.append(file_id)
            else:
                print(f"Warning: Missing annotation for {img_file}")
        
        return sorted(file_ids)  # Sort for consistent ordering
    
    # Get available files from each dataset
    print("Scanning datasets for available files...")
    chip2_files = set(get_available_files(chip2_path))
    chip4_files = set(get_available_files(chip4_path))
    chip2_preprocessed_files = set(get_available_files(chip2_preprocessed_path))
    chip4_preprocessed_files = set(get_available_files(chip4_preprocessed_path))
    
    print(f"Found {len(chip2_files)} files in chip2")
    print(f"Found {len(chip4_files)} files in chip4")
    print(f"Found {len(chip2_preprocessed_files)} files in chip2_preprocessed")
    print(f"Found {len(chip4_preprocessed_files)} files in chip4_preprocessed")
    
    # Find intersection to ensure we only use files that exist in both original and preprocessed versions
    # This prevents missing file errors during training
    chip2_valid_files = chip2_files.intersection(chip2_preprocessed_files)
    chip4_valid_files = chip4_files.intersection(chip4_preprocessed_files)
    
    print(f"Valid files (exist in both original and preprocessed):")
    print(f"  chip2: {len(chip2_valid_files)}")
    print(f"  chip4: {len(chip4_valid_files)}")
    
    # Ensure we have some valid files to work with
    if len(chip2_valid_files) == 0 and len(chip4_valid_files) == 0:
        raise ValueError("No valid files found. Check that file IDs match between original and preprocessed datasets.")
    
    # Convert to lists and shuffle for random splitting
    chip2_valid_list = list(chip2_valid_files)
    chip4_valid_list = list(chip4_valid_files)
    random.shuffle(chip2_valid_list)  # Randomize order before splitting
    random.shuffle(chip4_valid_list)
    
    def split_files(file_list: List[str], train_r: float, val_r: float, test_r: float) -> Tuple[List[str], List[str], List[str]]:
        """
        Split a list of files according to the given ratios.
        
        Args:
            file_list (List[str]): List of file IDs to split
            train_r (float): Training ratio
            val_r (float): Validation ratio
            test_r (float): Test ratio
            
        Returns:
            Tuple[List[str], List[str], List[str]]: (train_files, val_files, test_files)
        """
        n_total = len(file_list)
        n_train = int(n_total * train_r)
        n_val = int(n_total * val_r)
        # Remaining files go to test to handle rounding errors
        
        train_files = file_list[:n_train]
        val_files = file_list[n_train:n_train + n_val]
        test_files = file_list[n_train + n_val:]
        
        return train_files, val_files, test_files
    
    # Split each dataset independently to maintain proper ratios
    chip2_train, chip2_val, chip2_test = split_files(chip2_valid_list, train_ratio, val_ratio, test_ratio)
    chip4_train, chip4_val, chip4_test = split_files(chip4_valid_list, train_ratio, val_ratio, test_ratio)
    
    # Store split information in structured format
    split_info = {
        'train': {
            'chip2': chip2_train,
            'chip4': chip4_train
        },
        'val': {
            'chip2': chip2_val,
            'chip4': chip4_val
        },
        'test': {
            'chip2': chip2_test,
            'chip4': chip4_test
        }
    }
    
    # Print split statistics for verification
    print(f"\nSplit sizes:")
    for split in ['train', 'val', 'test']:
        total_files = len(split_info[split]['chip2']) + len(split_info[split]['chip4'])
        print(f"  {split}: {total_files} files (chip2: {len(split_info[split]['chip2'])}, chip4: {len(split_info[split]['chip4'])})")
    
    def copy_files_to_combined(file_ids: List[str], source_path: str, dest_images: Path, dest_annotations: Path, prefix: str):
        """
        Copy files to combined directory with unique naming to avoid conflicts.
        
        Args:
            file_ids (List[str]): List of file IDs to copy
            source_path (str): Source dataset path
            dest_images (Path): Destination images directory
            dest_annotations (Path): Destination annotations directory
            prefix (str): Prefix to add to filenames (e.g., 'chip2_', 'chip4_preprocessed_')
        """
        if not file_ids:
            return
            
        source_images = Path(source_path) / 'Images'
        source_annotations = Path(source_path) / 'Annotations'
        
        for file_id in file_ids:
            # Use prefix to avoid naming conflicts between datasets
            new_file_id = f"{prefix}_{file_id}"
            
            # Copy image file
            src_img = source_images / f'{file_id}.jpg'
            dst_img = dest_images / f'{new_file_id}.jpg'
            if src_img.exists():
                if copy_files:
                    shutil.copy2(src_img, dst_img)  # copy2 preserves metadata
            else:
                print(f"Warning: Missing image {src_img}")
            
            # Copy annotation file
            src_ann = source_annotations / f'{file_id}.json'
            dst_ann = dest_annotations / f'{new_file_id}.json'
            if src_ann.exists():
                if copy_files:
                    shutil.copy2(src_ann, dst_ann)
            else:
                print(f"Warning: Missing annotation {src_ann}")

    def copy_files_for_split(file_ids: List[str], source_path: str, dest_path: str, dataset_name: str, split_name: str):
        """
        Copy files for a specific split to individual dataset directories.
        
        Args:
            file_ids (List[str]): List of file IDs to copy
            source_path (str): Source dataset path
            dest_path (str): Base destination path
            dataset_name (str): Name of the dataset (e.g., 'chip2', 'chip4_preprocessed')
            split_name (str): Name of the split ('train', 'val', 'test')
        """
        if not file_ids:
            return
            
        source_images = Path(source_path) / 'Images'
        source_annotations = Path(source_path) / 'Annotations'
        dest_images = Path(dest_path) / split_name / dataset_name / 'Images'
        dest_annotations = Path(dest_path) / split_name / dataset_name / 'Annotations'
        
        for file_id in file_ids:
            # Copy image file (keep original filename in individual splits)
            src_img = source_images / f'{file_id}.jpg'
            dst_img = dest_images / f'{file_id}.jpg'
            if src_img.exists():
                if copy_files:
                    shutil.copy2(src_img, dst_img)
            else:
                print(f"Warning: Missing image {src_img}")
            
            # Copy annotation file
            src_ann = source_annotations / f'{file_id}.json'
            dst_ann = dest_annotations / f'{file_id}.json'
            if src_ann.exists():
                if copy_files:
                    shutil.copy2(src_ann, dst_ann)
            else:
                print(f"Warning: Missing annotation {src_ann}")
    
    # Copy files if requested (can be disabled for dry runs)
    if copy_files:
        print("\nCopying files...")
        
        # Define paths for all datasets
        dataset_paths = {
            'chip2': chip2_path,
            'chip4': chip4_path,
            'chip2_preprocessed': chip2_preprocessed_path,
            'chip4_preprocessed': chip4_preprocessed_path
        }
        
        # Process each split (train, val, test)
        for split in ['train', 'val', 'test']:
            print(f"  Processing {split} split...")
            
            # Create combined directories for SAM2 training (all datasets merged)
            combined_images = output_path / split / 'combined' / 'Images'
            combined_annotations = output_path / split / 'combined' / 'Annotations'
            combined_images.mkdir(parents=True, exist_ok=True)
            combined_annotations.mkdir(parents=True, exist_ok=True)
            
            # Copy files for each dataset type
            for dataset in ['chip2', 'chip4']:
                # Copy original datasets to separate directories
                copy_files_for_split(
                    split_info[split][dataset],
                    dataset_paths[dataset],
                    output_path,
                    dataset,
                    split
                )
                
                # Copy corresponding preprocessed datasets
                copy_files_for_split(
                    split_info[split][dataset],
                    dataset_paths[f'{dataset}_preprocessed'],
                    output_path,
                    f'{dataset}_preprocessed',
                    split
                )
                
                # Also copy to combined directory with unique naming
                # This creates a single directory with all datasets for SAM2 training
                copy_files_to_combined(
                    split_info[split][dataset],
                    dataset_paths[dataset],
                    combined_images,
                    combined_annotations,
                    prefix=dataset
                )
                
                copy_files_to_combined(
                    split_info[split][dataset],
                    dataset_paths[f'{dataset}_preprocessed'],
                    combined_images,
                    combined_annotations,
                    prefix=f'{dataset}_preprocessed'
                )
    
    # Save comprehensive metadata for reproducibility and analysis
    metadata = {
        'split_ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        },
        'seed': seed,
        'dataset_paths': {
            'chip2': str(chip2_path),
            'chip4': str(chip4_path),
            'chip2_preprocessed': str(chip2_preprocessed_path),
            'chip4_preprocessed': str(chip4_preprocessed_path)
        },
        'split_info': split_info,
        'statistics': {
            'total_files': {
                'chip2': len(chip2_valid_files),
                'chip4': len(chip4_valid_files)
            },
            'split_counts': {
                split: {
                    'chip2': len(split_info[split]['chip2']),
                    'chip4': len(split_info[split]['chip4']),
                    'total': len(split_info[split]['chip2']) + len(split_info[split]['chip4'])
                }
                for split in ['train', 'val', 'test']
            }
        }
    }
    
    # Save metadata as JSON for future reference
    with open(output_path / 'split_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create file lists for SAM2 training (useful for the file_list_txt parameter)
    print("Creating file lists for SAM2 training...")
    for split in ['train', 'val', 'test']:
        # Individual dataset file lists (for targeted training)
        for dataset in ['chip2', 'chip4', 'chip2_preprocessed', 'chip4_preprocessed']:
            file_list_path = output_path / f'{split}_{dataset}_files.txt'
            
            # Determine which file IDs to use
            if dataset.endswith('_preprocessed'):
                original_dataset = dataset.replace('_preprocessed', '')
                file_ids = split_info[split][original_dataset]
            else:
                file_ids = split_info[split][dataset]
            
            # Write file list (one file ID per line)
            with open(file_list_path, 'w') as f:
                for file_id in file_ids:
                    f.write(f'{file_id}\n')
        
        # Combined file list for SAM2 training (includes all datasets with prefixes)
        combined_file_list = output_path / f'{split}_combined_files.txt'
        with open(combined_file_list, 'w') as f:
            # Add all files with their prefixes to match combined directory structure
            for dataset in ['chip2', 'chip4']:
                for file_id in split_info[split][dataset]:
                    f.write(f'{dataset}_{file_id}\n')
                    f.write(f'{dataset}_preprocessed_{file_id}\n')
    
    # Print completion summary
    print(f"\n{'='*50}")
    print(f"Dataset split completed successfully!")
    print(f"{'='*50}")
    print(f"Output directory: {output_path}")
    print(f"Split metadata saved to: {output_path / 'split_metadata.json'}")
    print(f"File lists for SAM2 training saved as: {output_path / '*_files.txt'}")
    print(f"Combined datasets available in: {output_path / '*' / 'combined' / ''}")
    
    return split_info

