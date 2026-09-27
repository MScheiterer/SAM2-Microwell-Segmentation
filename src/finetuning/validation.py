import os
import re
import traceback
from pathlib import Path
from segmentation.model_evaluation import model_comparer
from segmentation.segmenter import *
from tqdm.notebook import tqdm
import pandas as pd
import matplotlib.pyplot as plt


def validate_checkpoints(image_dir: str, ann_dir: str, checkpoint_dir: str, output_dir: str) -> list[dict]:
    """
    Systematically evaluate all available model checkpoints to track training progress.
    
    This function discovers all checkpoint files in a directory, extracts epoch information,
    and evaluates each checkpoint using a standardized segmentation pipeline. It's designed
    to monitor model performance across training epochs and identify the best-performing
    checkpoint for deployment.
    
    Args:
        image_dir (str): Path to directory containing validation images
        ann_dir (str): Path to directory containing ground truth annotations
        checkpoint_dir (str): Path to directory containing model checkpoint files (.pt)
        output_dir (str): Path to directory where evaluation results will be saved
    
    Returns:
        List[Dict]: List of evaluation results, each containing:
            - results: Complete evaluation metrics from model_comparer
            - epoch: Epoch number of the evaluated checkpoint
    
    Raises:
        ValueError: If checkpoint directory doesn't exist
        FileNotFoundError: If required model config files are missing
        RuntimeError: If model loading or evaluation fails critically
    
    Directory Structure Expected:
        checkpoint_dir/
        ├── checkpoint_1.pt
        ├── checkpoint_5.pt
        ├── checkpoint_10.pt
        └── checkpoint.pt    # (skipped - ongoing training)
        
        output_dir/
        ├── val_results_1_summary.csv
        ├── val_results_5_summary.csv
        └── val_results_10_summary.csv
    
    Note:
        - Automatically skips checkpoints that have already been evaluated
        - Sorts checkpoints by epoch number for consistent evaluation order
        - Uses tqdm for progress tracking during evaluation
        - Saves detailed results for each checkpoint separately
        - Robust error handling allows evaluation to continue if individual checkpoints fail
        - Designed for SAM2 model architecture but adaptable to other segmentation models
    """
    
    # Convert to Path object for robust file operations
    checkpoint_dir = Path(checkpoint_dir)
    
    # Validate that checkpoint directory exists
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    # Discover all PyTorch checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return []
    
    # Extract epoch numbers and organize checkpoints
    checkpoints_with_epochs = []
    
    for checkpoint_file in checkpoint_files:
        filename = checkpoint_file.name
        
        # Extract epoch number using regex pattern
        # Matches patterns like: checkpoint_10.pt, checkpoint-15.pt, checkpoint10.pt
        epoch_pattern = r'checkpoint[_-]?(\d+)'
        
        epoch_num = None
        match = re.search(epoch_pattern, filename.lower())
        if match:
            epoch_num = int(match.group(1))
        
        if epoch_num is None:
            # Skip files without epoch numbers (e.g., checkpoint.pt from ongoing training)
            print(f"Skipping {filename}: no epoch number found")
            continue
        
        checkpoints_with_epochs.append((str(checkpoint_file), epoch_num))
    
    # Sort checkpoints by epoch number for logical evaluation order
    checkpoints_with_epochs.sort(key=lambda x: x[1])
    
    # Display discovered checkpoints for verification
    print(f"Found {len(checkpoints_with_epochs)} checkpoints with epoch numbers:")
    for checkpoint_path, epoch in checkpoints_with_epochs:
        print(f"  Epoch {epoch:2d}: {Path(checkpoint_path).name}")
    
    # Evaluate each checkpoint systematically
    results = []
    
    for i, (checkpoint_path, epoch_num) in enumerate(tqdm(checkpoints_with_epochs, desc="Evaluating checkpoints")):
        # Check if evaluation already exists to avoid duplicate work
        result_file_path = os.path.join(output_dir, f"val_results_{epoch_num}_summary.csv")
        
        if os.path.exists(result_file_path):
            print(f"Skipping epoch {epoch_num}: result already exists at {result_file_path}")
            continue
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Progress indicator for current evaluation
        print(f"\n[{i+1}/{len(checkpoints_with_epochs)}] Evaluating epoch {epoch_num}: {Path(checkpoint_path).name}")
        
        try:
            # Initialize SAM2 model with the current checkpoint
            # Using SAM2.1 Hiera Base+ configuration
            model_cfg_b_plus = "configs/sam2.1/sam2.1_hiera_b+.yaml"
            sam2 = SAM2Wrapper(
                config=model_cfg_b_plus, 
                checkpoint=checkpoint_path, 
                points_per_side=128  # Dense point grid for automatic segmentation
            )
            
            # Configure segmentation pipeline with filtering parameters
            # These parameters control post-processing and quality filtering
            segmenter = Segmenter(
                model=sam2,
                filter_config={
                    "circularity_threshold": 0.7,      # Filter non-circular objects
                    "border_thres": 2,                 # Border detection threshold
                    "max_groups": None,                # No limit on object groups
                    "eps_area": 40,                    # Minimum area for objects
                    "eps_coords_medium": 90,           # Coordinate epsilon for medium objects
                    "eps_coords_small": 70,            # Coordinate epsilon for small objects
                    "eps_coords_small_exclude": 20,    # Exclusion threshold for small objects
                    "rough_only": False,               # Use refined segmentation
                    "debug": True                      # Enable debug output
                }
            )
            
            # Run comprehensive model evaluation
            result = model_comparer(
                image_path=image_dir,              # Validation images
                annotation_path=ann_dir,           # Ground truth annotations
                segmenter=segmenter,               # Configured segmentation pipeline
                experiment_dir=output_dir,         # Results output directory
                experiment_name=f"val_results_{epoch_num}"  # Unique experiment identifier
            )
            
            # Store evaluation results with epoch information
            results.append({
                "results": result,      # Complete evaluation metrics
                "epoch": epoch_num      # Corresponding epoch number
            })
            
            print(f"Successfully evaluated epoch {epoch_num}")
            
            # Display key metrics for quick assessment
            if isinstance(result, dict) and 'mean_iou' in result:
                print(f"   Mean IoU: {result['mean_iou']:.3f}")
            
        except Exception as e:
            # Robust error handling - continue evaluation even if one checkpoint fails
            print(f"Error evaluating epoch {epoch_num}: {e}")
            print("Full stack trace:")
            print(traceback.format_exc())
            print("Continuing with next checkpoint...\n")
            continue
    
    # Summary of evaluation results
    print(f"\n{'='*60}")
    print(f"Checkpoint validation completed!")
    print(f"{'='*60}")
    print(f"Successfully evaluated: {len(results)} checkpoints")
    print(f"Results saved to: {output_dir}")
    
    # Display quick performance summary if results available
    if results and all('results' in r and isinstance(r['results'], dict) for r in results):
        print(f"\nPerformance Summary:")
        for result in sorted(results, key=lambda x: x['epoch']):
            epoch = result['epoch']
            metrics = result['results']
            if 'mean_iou' in metrics:
                print(f"  Epoch {epoch:2d}: IoU = {metrics['mean_iou']:.3f}")
    
    return results

def load_checkpoint_results(directory_path: str) -> pd.DataFrame:
    """
    Load all CSV files matching the pattern val_results_{epoch}_summary.csv and extract average metrics.
    
    This function systematically loads validation results from multiple checkpoint evaluations,
    extracting the average performance metrics for each epoch to enable training progress analysis.
    It's designed to work with the output from the validate_checkpoints function.
    
    Args:
        directory_path (str): Path to directory containing CSV files with validation results.
                             Expected files: val_results_1_summary.csv, val_results_5_summary.csv, etc.
    
    Returns:
        pd.DataFrame: DataFrame with epochs and corresponding average metrics. Columns include:
            - epoch: Training epoch number
            - precision_all: Overall precision across all object sizes
            - recall_all: Overall recall across all object sizes  
            - F1_all: Overall F1 score across all object sizes
            - average_IoU_all: Overall IoU across all object sizes
            - *_large, *_medium, *_small: Size-specific metrics
            - image_name: Set to 'average' for the summary row
    
    Raises:
        ValueError: If no matching CSV files found or no valid average rows found
        FileNotFoundError: If directory doesn't exist
        pd.errors.EmptyDataError: If CSV files are empty or corrupted
    
    Expected CSV Structure:
        Each CSV should contain rows for individual images plus an 'average' row:
        ```
        image_name,precision_all,recall_all,F1_all,average_IoU_all,...
        image1.jpg,0.85,0.82,0.83,0.78,...
        image2.jpg,0.88,0.85,0.86,0.81,...
        average,0.87,0.84,0.85,0.80,...
        ```
    """
    
    # Convert to Path object for robust file operations
    directory = Path(directory_path)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory_path}")
    
    # Find all CSV files matching the expected pattern
    csv_files = []
    
    for file in directory.glob("val_results_*_summary.csv"):
        # Extract epoch number using regex pattern
        # Matches: val_results_10_summary.csv -> epoch 10
        match = re.search(r'val_results_(\d+)_summary\.csv', file.name)
        if match:
            epoch = int(match.group(1))
            csv_files.append((epoch, file))
        else:
            print(f"Warning: Skipping file with unexpected format: {file.name}")
    
    if not csv_files:
        raise ValueError(f"No CSV files found matching pattern 'val_results_{{epoch}}_summary.csv' in {directory_path}")
    
    # Sort by epoch number for consistent processing
    csv_files.sort(key=lambda x: x[0])
    
    print(f"Found {len(csv_files)} checkpoint result files:")
    for epoch, file_path in csv_files:
        print(f"  Epoch {epoch:2d}: {file_path.name}")
    
    # Load each file and extract the average row
    results = []
    
    for epoch, file_path in csv_files:
        try:
            # Load CSV file
            df = pd.read_csv(file_path)
            
            if df.empty:
                print(f"Warning: Empty CSV file: {file_path.name}")
                continue
            
            # Find the "average" row (case insensitive search)
            # The average row contains summary statistics across all validation images
            avg_row = df[df['image_name'].str.lower() == 'average']
            
            if avg_row.empty:
                print(f"Warning: No 'average' row found in {file_path.name}")
                print(f"Available image_name values: {df['image_name'].unique()[:5]}...")
                continue
            
            # Extract the average row data and add epoch information
            avg_data = avg_row.iloc[0].copy()  # Get the first (should be only) average row
            avg_data['epoch'] = epoch
            results.append(avg_data)
            
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            continue
    
    if not results:
        raise ValueError("No valid average rows found in any CSV files")
    
    # Convert to DataFrame and sort by epoch
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('epoch').reset_index(drop=True)
    
    print(f"Successfully loaded data for {len(results_df)} epochs")
    
    return results_df


def plot_training_curves(results_df: pd.DataFrame, save_path: str = None, figsize: tuple = (15, 12)) -> None:
    """
    Plot comprehensive training curves for precision, recall, F1, and IoU metrics across all object sizes.
    
    Creates a 2x2 subplot layout showing training progression for different metrics and object sizes.
    Optionally saves both the complete figure and individual subplot files for publication use.
    
    Args:
        results_df (pd.DataFrame): DataFrame from load_checkpoint_results containing training metrics
        save_path (str, optional): Base path to save plots (without extension). If provided, saves as
                                  PDF, PNG, and SVG formats, plus individual subplot files.
        figsize (tuple, optional): Figure size as (width, height) in inches. Defaults to (15, 12).
    
    Returns:
        None: Displays the plot and optionally saves files
    
    Output Files (if save_path provided):
        - {save_path}.pdf/png/svg: Complete figure in multiple formats
        - individual_subplots/precision.pdf/svg: Individual precision plot
        - individual_subplots/recall.pdf/svg: Individual recall plot  
        - individual_subplots/f1_score.pdf/svg: Individual F1 score plot
        - individual_subplots/iou.pdf/svg: Individual IoU plot
    
    Plot Layout:
        ```
        ┌─────────────┬─────────────┐
        │ Precision   │ Recall      │
        │ vs Epoch    │ vs Epoch    │
        ├─────────────┼─────────────┤
        │ F1 Score    │ IoU         │
        │ vs Epoch    │ vs Epoch    │
        └─────────────┴─────────────┘
        ```
    
    Note:
        - Each subplot shows 4 lines: All sizes, Large, Medium, Small objects
        - Y-axis is limited to [0, 1.05] for better visualization of metric ranges
        - Uses distinct colors and line styles for each object size category
        - Grid is enabled for better readability
        - Individual subplot files are saved in SVG and PDF formats for flexibility
    """
    
    # Define the metrics to plot with their corresponding DataFrame columns
    metric_groups = {
        'Precision': ['precision_all', 'precision_large', 'precision_medium', 'precision_small'],
        'Recall': ['recall_all', 'recall_large', 'recall_medium', 'recall_small'],
        'F1 Score': ['F1_all', 'F1_large', 'F1_medium', 'F1_small'],
        'IoU': ['average_IoU_all', 'average_IoU_large', 'average_IoU_medium', 'average_IoU_small']
    }
    
    # Define consistent visual styling
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # blue, orange, green, red
    line_styles = ['-', '--', '-.', ':']  # solid, dashed, dash-dot, dotted
    size_labels = ['All', 'Large', 'Medium', 'Small']
    
    # Create the main figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Model Training Curves', fontsize=16, fontweight='bold')
    
    # Store subplot data for individual file saving
    subplot_data = {}
    
    # Create each subplot
    for idx, (metric_name, columns) in enumerate(metric_groups.items()):
        # Calculate subplot position
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Initialize storage for individual subplot saving
        subplot_data[metric_name] = {
            'columns': columns,
            'ax_data': {}
        }
        
        # Plot each size category (All, Large, Medium, Small)
        for i, column in enumerate(columns):
            if column in results_df.columns:
                # Check for missing data
                if results_df[column].isna().any():
                    print(f"Warning: Missing data in column {column}")
                
                # Plot the training curve
                line = ax.plot(results_df['epoch'], results_df[column], 
                              color=colors[i], linestyle=line_styles[i], 
                              marker='o', markersize=4, linewidth=2,
                              label=size_labels[i])
                
                # Store line data for individual subplot saving
                subplot_data[metric_name]['ax_data'][size_labels[i]] = {
                    'x': results_df['epoch'].values,
                    'y': results_df[column].values,
                    'color': colors[i],
                    'linestyle': line_styles[i],
                    'label': size_labels[i]
                }
            else:
                print(f"Warning: Column {column} not found in results DataFrame")
        
        # Customize subplot appearance
        ax.set_title(f'{metric_name} vs Epoch', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)  # Light grid for better readability
        ax.legend()
        
        # Set y-axis limits for better visualization (metrics are typically 0-1)
        if metric_name in ['Precision', 'Recall', 'F1 Score', 'IoU']:
            ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    # Save the complete figure if path provided
    if save_path:
        # Ensure output directory exists
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete figure in multiple formats for different use cases
        plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches='tight')  # High-quality print
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')  # Web/presentation
        plt.savefig(f"{save_path}.svg", bbox_inches='tight')           # Vector graphics
        print(f"Complete figure saved as: {save_path}.[pdf/png/svg]")
        
        # Save individual subplots for flexible use
        save_individual_subplots(subplot_data, save_path, results_df)
    
    plt.show()


def save_individual_subplots(subplot_data: dict, base_save_path: str, results_df: pd.DataFrame) -> None:
    """
    Save individual subplots as separate files for flexible use in publications.
    
    Creates individual plots for each metric type, allowing for flexible figure arrangement
    in papers, presentations, or reports. Each subplot is saved in both SVG and PDF formats.
    
    Args:
        subplot_data (dict): Data for each subplot extracted during main plotting
        base_save_path (str): Base path for saving files (used to determine output directory)
        results_df (pd.DataFrame): Original DataFrame for validation
    
    Creates:
        individual_subplots/precision.svg/pdf: Standalone precision plot
        individual_subplots/recall.svg/pdf: Standalone recall plot
        individual_subplots/f1_score.svg/pdf: Standalone F1 score plot
        individual_subplots/iou.svg/pdf: Standalone IoU plot
    """
    
    base_path = Path(base_save_path)
    individual_dir = base_path.parent / "individual_subplots"
    individual_dir.mkdir(parents=True, exist_ok=True)
    
    # Use consistent styling with main plot
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    line_styles = ['-', '--', '-.', ':']
    
    for metric_name, data in subplot_data.items():
        # Create individual figure for this metric
        fig_individual, ax_individual = plt.subplots(1, 1, figsize=(8, 6))
        
        # Recreate the plot for this specific metric
        size_labels = ['All', 'Large', 'Medium', 'Small']
        
        for i, size_label in enumerate(size_labels):
            if size_label in data['ax_data']:
                line_data = data['ax_data'][size_label]
                ax_individual.plot(line_data['x'], line_data['y'],
                                 color=line_data['color'], 
                                 linestyle=line_data['linestyle'],
                                 marker='o', markersize=4, linewidth=2,
                                 label=line_data['label'])
        
        # Customize individual plot to match main figure styling
        ax_individual.set_title(f'{metric_name} vs Epoch', fontweight='bold', fontsize=14)
        ax_individual.set_xlabel('Epoch', fontsize=12)
        ax_individual.set_ylabel(metric_name, fontsize=12)
        ax_individual.grid(True, alpha=0.3)
        ax_individual.legend()
        
        # Set consistent y-axis limits
        if metric_name in ['Precision', 'Recall', 'F1 Score', 'IoU']:
            ax_individual.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        # Save individual subplot in multiple formats
        metric_filename = metric_name.lower().replace(' ', '_')
        svg_filename = individual_dir / f"{metric_filename}.svg"
        pdf_filename = individual_dir / f"{metric_filename}.pdf"
        
        plt.savefig(svg_filename, bbox_inches='tight', format='svg')
        plt.savefig(pdf_filename, dpi=300, bbox_inches='tight', format='pdf')
        
        plt.close(fig_individual)  # Close to free memory
        
        print(f"Individual subplot saved: {svg_filename}")
        print(f"Individual subplot saved: {pdf_filename}")


def print_best_metrics(results_df: pd.DataFrame) -> None:
    """
    Print a summary of the best performance achieved for each metric across all epochs.
    
    Identifies and displays the epoch where each metric achieved its maximum value,
    helping to identify the best-performing checkpoint for deployment.
    
    Args:
        results_df (pd.DataFrame): DataFrame from load_checkpoint_results containing training metrics
    
    Returns:
        None: Prints formatted summary to console
    
    Output Format:
        ```
        Best Performance Summary:
        ==================================================
        PRECISION ALL  : Epoch  15 | Value: 0.8654
        RECALL ALL     : Epoch  12 | Value: 0.8432
        F1 ALL         : Epoch  14 | Value: 0.8541
        IOU ALL        : Epoch  13 | Value: 0.7892
        ```
    
    Example:
        >>> print_best_metrics(results_df)
        Best Performance Summary:
        ==================================================
        PRECISION ALL  : Epoch  25 | Value: 0.8756
        RECALL ALL     : Epoch  23 | Value: 0.8534
        F1 ALL         : Epoch  24 | Value: 0.8642
        IOU ALL        : Epoch  22 | Value: 0.8012
        
        >>> # Find the overall best epoch (by IoU)
        >>> best_iou_epoch = results_df.loc[results_df['average_IoU_all'].idxmax(), 'epoch']
        >>> print(f"Deploy checkpoint from epoch {best_iou_epoch}")
    """
    
    # Define metrics to analyze (focusing on overall performance)
    metrics = ['precision_all', 'recall_all', 'F1_all', 'average_IoU_all']
    
    print("Best Performance Summary:")
    print("=" * 50)
    
    for metric in metrics:
        if metric in results_df.columns:
            # Find the epoch with maximum value for this metric
            best_idx = results_df[metric].idxmax()
            best_epoch = results_df.loc[best_idx, 'epoch']
            best_value = results_df.loc[best_idx, metric]
            
            # Format metric name for display
            metric_display = metric.replace('_', ' ').replace('average ', '').upper()
            print(f"{metric_display:<15}: Epoch {best_epoch:3d} | Value: {best_value:.4f}")
        else:
            print(f"Warning: Metric {metric} not found in results")


def analyze_checkpoint_results(directory_path: str, save_plot: str = None) -> pd.DataFrame:
    """
    Complete pipeline to analyze checkpoint validation results with visualization and summary.
    
    This is the main function that orchestrates the entire analysis workflow:
    1. Loads checkpoint results from CSV files
    2. Creates comprehensive training curve visualizations  
    3. Identifies and reports best-performing epochs
    4. Returns processed data for further analysis
    
    Args:
        directory_path (str): Path to directory containing validation result CSV files
        save_plot (str, optional): Base path to save visualization plots
        
    Returns:
        pd.DataFrame: Processed results DataFrame for additional analysis
    
    Workflow:
        1. **Data Loading**: Discovers and loads all checkpoint result files
        2. **Validation**: Checks data integrity and reports any issues
        3. **Visualization**: Creates training curves for all metrics
        4. **Analysis**: Identifies best-performing epochs per metric
        5. **Reporting**: Displays summary statistics and recommendationss
    
    Output Files (if save_plot provided):
        - Training curve plots in PDF, PNG, SVG formats
        - Individual subplot files for publication use
        - Console output with performance summary
    
    Note:
        - Automatically handles missing files or corrupted data
        - Provides warnings for any data quality issues
        - Returns DataFrame for additional custom analysis
        - Integrates seamlessly with checkpoint validation workflow
    """
    
    print(f"Loading checkpoint results from: {directory_path}")
    
    try:
        # Load and validate the checkpoint results
        results_df = load_checkpoint_results(directory_path)
        
        print(f"Loaded {len(results_df)} checkpoint results")
        print(f"Epochs: {results_df['epoch'].min()} - {results_df['epoch'].max()}")
        
        # Check for data quality issues
        total_metrics = len([col for col in results_df.columns if col != 'epoch' and col != 'image_name'])
        missing_data = results_df.isnull().sum().sum()
        if missing_data > 0:
            print(f"Warning: Found {missing_data} missing values in the data")
        
        # Create comprehensive training curve visualizations
        print("\nGenerating training curves...")
        plot_training_curves(results_df, save_path=save_plot)
        
        # Analyze and report best performance
        print("\nAnalyzing best performance...")
        print_best_metrics(results_df)
        
        # Additional insights
        print(f"\nTraining Progress Summary:")
        print(f"- Total epochs analyzed: {len(results_df)}")
        print(f"- Metrics tracked: {total_metrics}")
        
        # Calculate overall trends
        if 'average_IoU_all' in results_df.columns and len(results_df) > 1:
            iou_trend = results_df['average_IoU_all'].iloc[-1] - results_df['average_IoU_all'].iloc[0]
            trend_direction = "improving" if iou_trend > 0 else "declining" if iou_trend < 0 else "stable"
            print(f"- Overall IoU trend: {trend_direction} ({iou_trend:+.4f})")
        
        return results_df
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise
