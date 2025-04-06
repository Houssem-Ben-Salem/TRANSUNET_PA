import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import binary_dilation, binary_erosion
import time

# Import necessary modules
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import ModelWrapper, ValidationTransform, TestTimeAugmentation
from improved_dataset import PAFocusedDataset

# Create logger
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def calculate_metrics(pred, gt):
    """Calculate various segmentation metrics
    
    Args:
        pred (np.ndarray): Binary prediction mask
        gt (np.ndarray): Binary ground truth mask
        
    Returns:
        dict: Dictionary containing various metrics
    """
    # Ensure binary masks
    pred = (pred > 0).astype(np.float32)
    gt = (gt > 0).astype(np.float32)
    
    # Calculate intersection and union
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    
    # Dice coefficient (F1 score)
    dice = (2. * intersection) / (union + 1e-6)
    
    # Precision and recall
    if np.sum(pred) > 0:
        precision = intersection / (np.sum(pred) + 1e-6)
    else:
        precision = 0.0
    
    if np.sum(gt) > 0:
        recall = intersection / (np.sum(gt) + 1e-6)
    else:
        recall = 0.0
    
    # Jaccard index (IoU)
    iou = intersection / (union - intersection + 1e-6)
    
    # Volumetric overlap error (1 - IoU)
    voe = 1.0 - iou
    
    # Calculate Hausdorff distance if there's content in both masks
    hd95 = 0.0
    if np.sum(pred) > 0 and np.sum(gt) > 0:
        try:
            from medpy.metric.binary import hd95 as medpy_hd95
            hd95 = medpy_hd95(pred, gt)
        except ImportError:
            # If medpy not available, use a placeholder
            hd95 = -1.0
    
    # Calculate boundary error metrics
    # Dilate and erode ground truth to get boundary
    gt_dilated = binary_dilation(gt, iterations=1)
    gt_eroded = binary_erosion(gt, iterations=1)
    gt_boundary = gt_dilated.astype(int) - gt_eroded.astype(int)
    
    # Do the same for prediction
    pred_dilated = binary_dilation(pred, iterations=1)
    pred_eroded = binary_erosion(pred, iterations=1)
    pred_boundary = pred_dilated.astype(int) - pred_eroded.astype(int)
    
    # Calculate boundary precision and recall
    boundary_intersection = np.sum(pred_boundary * gt_boundary)
    boundary_precision = boundary_intersection / (np.sum(pred_boundary) + 1e-6)
    boundary_recall = boundary_intersection / (np.sum(gt_boundary) + 1e-6)
    
    # Calculate specificity (true negative rate)
    true_neg = np.sum((1 - pred) * (1 - gt))
    specificity = true_neg / (np.sum(1 - gt) + 1e-6)
    
    # F1 score for boundary (useful for thin structures like PA)
    boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall + 1e-6)
    
    return {
        'dice': dice,
        'precision': precision,
        'recall': recall, 
        'iou': iou,
        'voe': voe,
        'hd95': hd95,
        'boundary_precision': boundary_precision,
        'boundary_recall': boundary_recall,
        'boundary_f1': boundary_f1,
        'specificity': specificity
    }

def visualize_prediction(image, prediction, ground_truth, case_id, slice_idx, output_dir):
    """
    Visualize prediction against ground truth
    
    Args:
        image (np.ndarray): Input image
        prediction (np.ndarray): Prediction mask
        ground_truth (np.ndarray): Ground truth mask
        case_id (str): Case identifier
        slice_idx (int): Slice index
        output_dir (str): Output directory for saving visualizations
    """
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Normalize image for visualization
    if image.min() != image.max():
        img_norm = (image - image.min()) / (image.max() - image.min())
    else:
        img_norm = image
    
    # Plot image
    axes[0].imshow(img_norm, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Plot prediction
    axes[1].imshow(img_norm, cmap='gray')
    mask = np.ma.masked_where(prediction == 0, prediction)
    axes[1].imshow(mask, cmap='cool', alpha=0.6)
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # Plot ground truth
    axes[2].imshow(img_norm, cmap='gray')
    mask = np.ma.masked_where(ground_truth == 0, ground_truth)
    axes[2].imshow(mask, cmap='hot', alpha=0.6)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    # Add metrics to the title
    metrics = calculate_metrics(prediction, ground_truth)
    plt.suptitle(f"Case: {case_id}, Slice: {slice_idx}\nDice: {metrics['dice']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    
    # Save figure
    viz_dir = os.path.join(output_dir, 'slice_visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    plt.savefig(os.path.join(viz_dir, f'{case_id}_slice{slice_idx}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

def reconstruct_3d_volume(predictions, original_shape):
    """
    Reconstruct a 3D volume from slice predictions
    
    Args:
        predictions (dict): Dictionary mapping slice indices to prediction masks
        original_shape (tuple): Original 3D volume shape
        
    Returns:
        np.ndarray: Reconstructed 3D volume
    """
    # Create an empty volume with the original shape
    volume = np.zeros(original_shape, dtype=np.uint8)
    
    # Fill in the slices
    for slice_idx, pred_mask in predictions.items():
        # Resize prediction to original slice size if needed
        if pred_mask.shape != original_shape[1:]:
            from skimage.transform import resize
            pred_mask = resize(pred_mask, original_shape[1:], order=0, preserve_range=True)
        
        # Add to volume
        volume[slice_idx] = pred_mask
    
    return volume

def save_nifti(volume, case_id, output_dir, reference_nifti=None):
    """
    Save a volume as NIFTI file with metadata from a reference image
    
    Args:
        volume (np.ndarray): 3D volume to save
        case_id (str): Case identifier
        output_dir (str): Output directory
        reference_nifti (sitk.Image, optional): Reference NIFTI to copy metadata from
    """
    pred_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    output_path = os.path.join(pred_dir, f'{case_id}_pred.nii.gz')
    
    # If reference NIFTI is provided, copy metadata
    if reference_nifti is not None:
        # Create a new image with the prediction data but metadata from reference
        pred_sitk = sitk.GetImageFromArray(volume.astype(np.uint8))
        pred_sitk.CopyInformation(reference_nifti)
        sitk.WriteImage(pred_sitk, output_path)
    else:
        # Create a simple NIFTI file without specific metadata
        pred_sitk = sitk.GetImageFromArray(volume.astype(np.uint8))
        sitk.WriteImage(pred_sitk, output_path)
    
    logger.info(f"Saved prediction for {case_id} to {output_path}")

def create_visualizations(results_df, output_dir):
    """
    Create visualizations for metric distributions
    
    Args:
        results_df (pd.DataFrame): DataFrame with results
        output_dir (str): Output directory
    """
    # Remove 'AVERAGE' row for visualizations
    df_viz = results_df[results_df['case_id'] != 'AVERAGE'].copy()
    
    # Create directory for visualizations
    viz_dir = os.path.join(output_dir, 'metric_visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Plot 1: Dice score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df_viz['dice'], kde=True, bins=15)
    plt.title('Distribution of Dice Scores Across Test Cases', fontsize=14)
    plt.xlabel('Dice Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.axvline(df_viz['dice'].mean(), color='r', linestyle='--', 
                label=f'Mean: {df_viz["dice"].mean():.4f}')
    plt.axvline(df_viz['dice'].median(), color='g', linestyle='--', 
                label=f'Median: {df_viz["dice"].median():.4f}')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'dice_distribution.png'), dpi=200)
    plt.close()
    
    # Plot 2: Precision vs recall scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(df_viz['precision'], df_viz['recall'], alpha=0.7, s=60)
    
    # Add case_id labels to points
    for idx, row in df_viz.iterrows():
        plt.annotate(row['case_id'], 
                    (row['precision'], row['recall']),
                    fontsize=8, 
                    alpha=0.7,
                    xytext=(5, 5),
                    textcoords='offset points')
    
    plt.title('Precision vs Recall for Each Test Case', fontsize=14)
    plt.xlabel('Precision', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.grid(True)
    plt.axvline(0.5, color='r', linestyle='--', alpha=0.3)
    plt.axhline(0.5, color='r', linestyle='--', alpha=0.3)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'precision_recall_scatter.png'), dpi=200)
    plt.close()
    
    # Plot 3: Performance metrics by case (sorted)
    plt.figure(figsize=(12, 8))
    df_sorted = df_viz.sort_values('dice', ascending=False).reset_index(drop=True)
    
    plt.bar(df_sorted.index, df_sorted['dice'], alpha=0.7, label='Dice')
    plt.plot(df_sorted.index, df_sorted['precision'], 'ro-', alpha=0.7, label='Precision')
    plt.plot(df_sorted.index, df_sorted['recall'], 'go-', alpha=0.7, label='Recall')
    
    plt.title('Performance Metrics by Case (Sorted by Dice Score)', fontsize=14)
    plt.xlabel('Case Index (Sorted)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(df_sorted.index, df_sorted['case_id'], rotation=90)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'metrics_by_case.png'), dpi=200)
    plt.close()
    
    # Plot 4: Volume comparison (predicted vs ground truth)
    plt.figure(figsize=(10, 8))
    plt.scatter(df_viz['gt_volume_ml'], df_viz['pred_volume_ml'], alpha=0.7, s=60)
    
    # Add case_id labels
    for idx, row in df_viz.iterrows():
        plt.annotate(row['case_id'], 
                    (row['gt_volume_ml'], row['pred_volume_ml']),
                    fontsize=8, 
                    alpha=0.7,
                    xytext=(5, 5),
                    textcoords='offset points')
    
    # Add perfect prediction line
    min_vol = min(df_viz['gt_volume_ml'].min(), df_viz['pred_volume_ml'].min())
    max_vol = max(df_viz['gt_volume_ml'].max(), df_viz['pred_volume_ml'].max())
    padding = (max_vol - min_vol) * 0.1
    plt.plot([min_vol-padding, max_vol+padding], [min_vol-padding, max_vol+padding], 
             'k--', alpha=0.5, label='Perfect Prediction')
    
    plt.title('Ground Truth vs Predicted Volume Comparison', fontsize=14)
    plt.xlabel('Ground Truth Volume (ml)', fontsize=12)
    plt.ylabel('Predicted Volume (ml)', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'volume_comparison.png'), dpi=200)
    plt.close()
    
    # Plot 5: Summary of all metrics
    metrics_to_plot = ['dice', 'precision', 'recall', 'iou', 'specificity', 
                      'boundary_f1', 'boundary_precision', 'boundary_recall']
    
    plt.figure(figsize=(12, 8))
    avg_metrics = df_viz[metrics_to_plot].mean()
    std_metrics = df_viz[metrics_to_plot].std()
    
    x = np.arange(len(metrics_to_plot))
    width = 0.7
    
    # Plot bars for mean values
    bars = plt.bar(x, avg_metrics, width, yerr=std_metrics, 
                  capsize=5, alpha=0.7, ecolor='black')
    
    # Customize the plot
    plt.ylabel('Score', fontsize=12)
    plt.title('Average Performance Across All Metrics', fontsize=14)
    plt.xticks(x, [m.replace('_', ' ').title() for m in metrics_to_plot], rotation=45)
    plt.ylim(0, 1.05)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', fontsize=9)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'metrics_summary.png'), dpi=200)
    plt.close()
    
    # Create a comprehensive summary figure
    create_summary_figure(df_viz, viz_dir)
    
    logger.info(f"Saved metric visualizations to {viz_dir}")

def create_summary_figure(df, output_dir):
    """
    Create a summary figure with multiple visualizations combined
    
    Args:
        df (pd.DataFrame): DataFrame with results
        output_dir (str): Output directory
    """
    fig = plt.figure(figsize=(22, 16))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Plot 1: Dice score distribution
    ax1 = fig.add_subplot(2, 2, 1)
    sns.histplot(df['dice'], kde=True, bins=15, ax=ax1)
    ax1.set_title('Distribution of Dice Scores', fontsize=14)
    ax1.set_xlabel('Dice Score', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.axvline(df['dice'].mean(), color='r', linestyle='--', 
                label=f'Mean: {df["dice"].mean():.4f}')
    ax1.axvline(df['dice'].median(), color='g', linestyle='--', 
                label=f'Median: {df["dice"].median():.4f}')
    ax1.legend(fontsize=10)
    
    # Plot 2: Precision vs recall
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(df['precision'], df['recall'], alpha=0.7, s=60)
    
    # Add case labels
    for idx, row in df.iterrows():
        ax2.annotate(row['case_id'], 
                    (row['precision'], row['recall']),
                    fontsize=8, 
                    alpha=0.7,
                    xytext=(5, 5),
                    textcoords='offset points')
    
    ax2.set_title('Precision vs Recall', fontsize=14)
    ax2.set_xlabel('Precision', fontsize=12)
    ax2.set_ylabel('Recall', fontsize=12)
    ax2.grid(True)
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    
    # Plot 3: Performance metrics by case
    ax3 = fig.add_subplot(2, 2, 3)
    df_sorted = df.sort_values('dice', ascending=False).reset_index(drop=True)
    
    # Use only a subset of cases if there are too many
    max_cases_to_show = 15
    if len(df_sorted) > max_cases_to_show:
        df_plot = pd.concat([
            df_sorted.head(max_cases_to_show//2),  # Top performers
            df_sorted.tail(max_cases_to_show//2)   # Bottom performers
        ])
    else:
        df_plot = df_sorted
    
    x = np.arange(len(df_plot))
    width = 0.25
    
    ax3.bar(x - width, df_plot['dice'], width, label='Dice', alpha=0.7)
    ax3.bar(x, df_plot['precision'], width, label='Precision', alpha=0.7)
    ax3.bar(x + width, df_plot['recall'], width, label='Recall', alpha=0.7)
    
    ax3.set_title('Performance by Case', fontsize=14)
    ax3.set_xlabel('Case ID', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_plot['case_id'], rotation=90, fontsize=8)
    ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Volume comparison
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(df['gt_volume_ml'], df['pred_volume_ml'], alpha=0.7, s=60)
    
    # Perfect prediction line
    min_vol = min(df['gt_volume_ml'].min(), df['pred_volume_ml'].min())
    max_vol = max(df['gt_volume_ml'].max(), df['pred_volume_ml'].max())
    padding = (max_vol - min_vol) * 0.1
    ax4.plot([min_vol-padding, max_vol+padding], [min_vol-padding, max_vol+padding], 
             'k--', alpha=0.5, label='Perfect prediction')
    
    # Add percentage difference annotation
    for idx, row in df.iterrows():
        pct_diff = ((row['pred_volume_ml'] - row['gt_volume_ml']) / row['gt_volume_ml']) * 100
        color = 'red' if abs(pct_diff) > 20 else 'black'
        ax4.annotate(f"{row['case_id']}\n({pct_diff:.1f}%)", 
                    (row['gt_volume_ml'], row['pred_volume_ml']),
                    fontsize=7, 
                    color=color,
                    alpha=0.8,
                    xytext=(5, 5),
                    textcoords='offset points')
    
    ax4.set_title('Ground Truth vs Predicted Volume', fontsize=14)
    ax4.set_xlabel('Ground Truth Volume (ml)', fontsize=12)
    ax4.set_ylabel('Predicted Volume (ml)', fontsize=12)
    ax4.grid(True)
    ax4.legend(fontsize=10)
    
    # Add summary statistics as text
    plt.figtext(0.5, 0.01, 
                f"Summary Statistics (N={len(df)})\n"
                f"Mean Dice: {df['dice'].mean():.4f} ± {df['dice'].std():.4f}\n"
                f"Mean IoU: {df['iou'].mean():.4f} ± {df['iou'].std():.4f}\n"
                f"Mean Precision: {df['precision'].mean():.4f} ± {df['precision'].std():.4f}\n"
                f"Mean Recall: {df['recall'].mean():.4f} ± {df['recall'].std():.4f}\n"
                f"Mean Volume Error: {(df['pred_volume_ml'] - df['gt_volume_ml']).mean():.2f} ± {(df['pred_volume_ml'] - df['gt_volume_ml']).std():.2f} ml",
                ha='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(os.path.join(output_dir, 'performance_summary.png'), dpi=200, bbox_inches='tight')
    plt.close()

def test_model(args):
    """
    Test a trained model on the test dataset
    
    Args:
        args: Command-line arguments
        
    Returns:
        pd.DataFrame: DataFrame with per-case statistics
    """
    # Start timer
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logger to also write to file
    log_file = os.path.join(args.output_dir, 'test_log.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting testing with args: {vars(args)}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load model configuration
    dataset_config = {
        'Parse': {
            'root_path': './DATA',
            'list_dir': './lists/lists_Parse',
            'num_classes': 2,  # Background and PA
        },
    }
    
    # Update config from dataset configuration
    args.num_classes = dataset_config[args.dataset]['num_classes']
    args.root_path = dataset_config[args.dataset]['root_path']
    args.list_dir = dataset_config[args.dataset]['list_dir']
    
    # Initialize model
    logger.info(f"Initializing model: {args.vit_name}")
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), 
                                  int(args.img_size / args.vit_patches_size))
    
    # Create model
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device)
    
    # Wrap model to handle input channel conversion
    net = ModelWrapper(net)
    
    # Load model from checkpoint
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Check if checkpoint is for wrapped model
            if isinstance(net, ModelWrapper):
                try:
                    # Try loading into wrapped model
                    net.model.load_state_dict(checkpoint['model_state_dict'])
                except Exception as e:
                    logger.warning(f"Failed to load directly into wrapped model: {str(e)}. Trying alternate approach.")
                    # Try with model prefix
                    state_dict = {}
                    for k, v in checkpoint['model_state_dict'].items():
                        if k.startswith('model.'):
                            state_dict[k[6:]] = v  # Remove 'model.' prefix
                        else:
                            state_dict[k] = v
                    net.model.load_state_dict(state_dict)
            else:
                net.load_state_dict(checkpoint['model_state_dict'])
            
            # Get additional info if available
            if 'epoch' in checkpoint:
                logger.info(f"Checkpoint from epoch: {checkpoint['epoch']}")
            if 'best_performance' in checkpoint:
                logger.info(f"Checkpoint best performance: {checkpoint['best_performance']:.4f}")
        else:
            # Just load state dict
            if isinstance(net, ModelWrapper):
                try:
                    net.model.load_state_dict(checkpoint)
                except:
                    # Try to update keys
                    state_dict = {}
                    for k, v in checkpoint.items():
                        if k.startswith('model.'):
                            state_dict[k[6:]] = v  # Remove 'model.' prefix
                        else:
                            state_dict[k] = v
                    net.model.load_state_dict(state_dict)
            else:
                net.load_state_dict(checkpoint)
            
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}")
        raise
    
    # Set model to evaluation mode
    net.eval()
    
    # Initialize test-time augmentation if enabled
    if args.use_tta:
        logger.info(f"Using test-time augmentation with {args.tta_num} augmentations")
        tta = TestTimeAugmentation(num_augmentations=args.tta_num)
    
    # Create list of test cases
    with open(os.path.join(args.list_dir, 'test_vol.txt'), 'r') as f:
        test_cases = [line.strip() for line in f.readlines()]
    
    logger.info(f"Found {len(test_cases)} test cases")
    
    # Initialize results storage
    case_results = []
    
    # Process each test case
    for case_id in tqdm(test_cases, desc="Processing test cases"):
        # Get paths for this case
        img_path = os.path.join(args.root_path, case_id, "image", f"{case_id}.nii.gz")
        lab_path = os.path.join(args.root_path, case_id, "label", f"{case_id}.nii.gz")
        
        # Check if files exist
        if not os.path.exists(img_path) or not os.path.exists(lab_path):
            logger.warning(f"Missing files for {case_id}, skipping")
            continue
        
        # Load volumes
        img_sitk = sitk.ReadImage(img_path)
        lab_sitk = sitk.ReadImage(lab_path)
        
        img_data = sitk.GetArrayFromImage(img_sitk)
        lab_data = sitk.GetArrayFromImage(lab_sitk)
        
        # Get spacing for volume calculation
        spacing = img_sitk.GetSpacing()
        if len(spacing) == 3:
            voxel_volume = np.prod(spacing) / 1000  # mm³ to ml
        else:
            # Default to 1 mm spacing if not available
            voxel_volume = 0.001  # 1 mm³ = 0.001 ml
            
        # Store original shape for reconstruction
        original_shape = img_data.shape
        
        # Store predictions for this case
        case_predictions = {}
        slice_metrics = []
        
        # Process each slice
        for slice_idx in range(img_data.shape[0]):
            # Skip slices with no content in ground truth if enabled
            if args.skip_empty_slices and np.sum(lab_data[slice_idx]) == 0:
                continue
            
            # Extract slice
            img_slice = img_data[slice_idx]
            lab_slice = lab_data[slice_idx]
            
            # Normalize image to 0-1
            if img_slice.max() != img_slice.min():
                img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
            
            # Prepare for model
            img_tensor = torch.from_numpy(img_slice).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            
            # Resize to model input size
            img_tensor = F.interpolate(img_tensor, size=(args.img_size, args.img_size), 
                                       mode='bilinear', align_corners=True)
            
            # Move to device
            img_tensor = img_tensor.to(device)
            
            # Get prediction
            with torch.no_grad():
                if args.use_tta:
                    # Apply test-time augmentation
                    outputs_soft = tta(net, img_tensor)
                else:
                    # Standard forward pass
                    outputs = net(img_tensor)
                    outputs_soft = torch.softmax(outputs, dim=1)
                
                # Get the class prediction
                outputs_pred = torch.argmax(outputs_soft, dim=1).squeeze().cpu().numpy()
            
            # Resize back to original shape
            if outputs_pred.shape != lab_slice.shape:
                from skimage.transform import resize
                outputs_pred = resize(outputs_pred, lab_slice.shape, order=0, preserve_range=True)
            
            # Ensure binary mask
            outputs_pred = (outputs_pred > 0.5).astype(np.uint8)
            
            # Store prediction
            case_predictions[slice_idx] = outputs_pred
            
            # Calculate metrics for this slice
            slice_metrics.append(calculate_metrics(outputs_pred, lab_slice))
            
            # Visualize some predictions (e.g., every 10th slice with PA or randomly)
            if (slice_idx % 10 == 0) or (np.sum(lab_slice) > 0 and np.random.random() < 0.2):
                visualize_prediction(img_slice, outputs_pred, lab_slice, case_id, slice_idx, args.output_dir)
        
        # Reconstruct 3D volume from slice predictions
        reconstructed_volume = reconstruct_3d_volume(case_predictions, original_shape)
        
        # Save as NIFTI
        save_nifti(reconstructed_volume, case_id, args.output_dir, reference_nifti=img_sitk)
        
        # Calculate overall case metrics using 3D volumes
        case_3d_metrics = calculate_metrics(reconstructed_volume, lab_data)
        
        # Calculate average slice metrics
        avg_slice_metrics = {}
        for metric in slice_metrics[0].keys():
            values = [m[metric] for m in slice_metrics if m[metric] >= 0]
            avg_slice_metrics[f'avg_slice_{metric}'] = np.mean(values) if values else 0
        
        # Calculate volumes
        gt_volume_ml = np.sum(lab_data) * voxel_volume
        pred_volume_ml = np.sum(reconstructed_volume) * voxel_volume
        
        # Combine all metrics for this case
        combined_metrics = {
            'case_id': case_id,
            **{f'3d_{k}': v for k, v in case_3d_metrics.items()},  # 3D metrics
            **avg_slice_metrics,  # Average slice metrics
            'num_slices': len(slice_metrics),
            'pa_slices': sum(1 for slice_idx in range(original_shape[0]) if np.sum(lab_data[slice_idx]) > 0),
            'gt_volume_ml': gt_volume_ml,
            'pred_volume_ml': pred_volume_ml,
            'volume_diff_ml': pred_volume_ml - gt_volume_ml,
            'volume_diff_percent': ((pred_volume_ml - gt_volume_ml) / max(gt_volume_ml, 1e-6)) * 100
        }
        
        # Add to results
        case_results.append(combined_metrics)
        
        # Log case results
        logger.info(f"\nCase {case_id} results:")
        logger.info(f"3D Dice: {combined_metrics['3d_dice']:.4f}")
        logger.info(f"3D Precision: {combined_metrics['3d_precision']:.4f}")
        logger.info(f"3D Recall: {combined_metrics['3d_recall']:.4f}")
        logger.info(f"Volume: GT={gt_volume_ml:.2f}ml, Pred={pred_volume_ml:.2f}ml, Diff={combined_metrics['volume_diff_ml']:.2f}ml ({combined_metrics['volume_diff_percent']:.1f}%)")
    
    # Skip further processing if no results
    if not case_results:
        logger.error("No valid test cases processed. Check your data paths.")
        return None
    
    # Create DataFrame for all results
    results_df = pd.DataFrame(case_results)
    
    # Rename some columns for easier reference
    rename_map = {
        '3d_dice': 'dice',
        '3d_precision': 'precision',
        '3d_recall': 'recall',
        '3d_iou': 'iou',
        '3d_specificity': 'specificity',
        '3d_boundary_f1': 'boundary_f1'
    }
    results_df = results_df.rename(columns=rename_map)
    
    # Calculate average metrics across all cases
    avg_metrics = results_df.drop(columns=['case_id']).mean(numeric_only=True)
    std_metrics = results_df.drop(columns=['case_id']).std(numeric_only=True)
    
    # Add a row for average metrics
    avg_row = {'case_id': 'AVERAGE', **{col: avg_metrics[col] for col in avg_metrics.index}}
    std_row = {'case_id': 'STD', **{col: std_metrics[col] for col in std_metrics.index}}
    
    results_df = pd.concat([results_df, pd.DataFrame([avg_row, std_row])], ignore_index=True)
    
    # Save detailed results to CSV
    results_path = os.path.join(args.output_dir, 'test_results_detailed.csv')
    results_df.to_csv(results_path, index=False)
    
    # Create a simplified version with key metrics
    key_metrics = ['case_id', 'dice', 'precision', 'recall', 'iou', 'boundary_f1', 
                  'gt_volume_ml', 'pred_volume_ml', 'volume_diff_ml', 'volume_diff_percent']
    simple_results = results_df[key_metrics].copy()
    
    # Save simplified results
    simple_path = os.path.join(args.output_dir, 'test_results.csv')
    simple_results.to_csv(simple_path, index=False)
    
    logger.info(f"\nResults saved to {results_path} and {simple_path}")
    
    # Create visualizations of metric distributions
    create_visualizations(results_df[results_df['case_id'] != 'STD'], args.output_dir)
    
    # Print summary statistics
    logger.info("\n=== SUMMARY STATISTICS ===")
    logger.info(f"Number of test cases: {len(results_df) - 2}")  # Subtract AVERAGE and STD rows
    logger.info(f"Average Dice score: {avg_metrics['dice']:.4f} ± {std_metrics['dice']:.4f}")
    logger.info(f"Average IoU: {avg_metrics['iou']:.4f} ± {std_metrics['iou']:.4f}")
    logger.info(f"Average Precision: {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    logger.info(f"Average Recall: {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    logger.info(f"Average Volume Error: {avg_metrics['volume_diff_ml']:.2f} ± {std_metrics['volume_diff_ml']:.2f} ml")
    logger.info(f"Average Volume Error Percentage: {avg_metrics['volume_diff_percent']:.2f} ± {std_metrics['volume_diff_percent']:.2f}%")
    logger.info(f"Testing completed in {(time.time() - start_time)/60:.2f} minutes")
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test TransUNet on segmentation dataset')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to the model checkpoint to evaluate')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                       help='Select the Vision Transformer model architecture')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input patch size of network input')
    parser.add_argument('--n_skip', type=int, default=3,
                       help='Using number of skip-connect')
    parser.add_argument('--vit_patches_size', type=int, default=16,
                       help='ViT patches size')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='Parse',
                       help='Dataset name')
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Parse',
                       help='List directory with test_vol.txt')
    parser.add_argument('--root_path', type=str, default='./DATA',
                       help='Root directory for data')
    
    # Test parameters
    parser.add_argument('--output_dir', type=str, default='./test_results',
                       help='Directory to save results and visualizations')
    parser.add_argument('--seed', type=int, default=1234, 
                       help='Random seed for reproducibility')
    parser.add_argument('--skip_empty_slices', action='store_true',
                       help='Skip slices with no content in ground truth for speed')
    parser.add_argument('--use_tta', action='store_true',
                       help='Use test-time augmentation for improved prediction')
    parser.add_argument('--tta_num', type=int, default=4,
                       help='Number of test-time augmentations')
    
    args = parser.parse_args()
    
    # Run the test
    results = test_model(args)
    
    # Final message
    if results is not None:
        print("\nTesting completed successfully. Results saved to:", args.output_dir)
        print("To view the results, check the following files:")
        print(f"1. {os.path.join(args.output_dir, 'test_results.csv')} - Summary statistics")
        print(f"2. {os.path.join(args.output_dir, 'metric_visualizations/performance_summary.png')} - Visual summary")
        print(f"3. {os.path.join(args.output_dir, 'predictions/')} - Predicted segmentation masks")
    else:
        print("\nTesting failed. Check the logs for more information.")