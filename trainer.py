import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.transforms.functional as TF
from test_time_aug import TestTimeAugmentation

# Import our custom modules directly
from improved_dataset import PAFocusedDataset
from augmentations import EnhancedRandomGenerator
from losses import CombinedLoss, DiceLoss, TverskyLoss

# Custom learning rate scheduler with warmup
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

# Add ModelWrapper to handle input channel conversion
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        # Ensure input has correct shape [B, C, H, W]
        # If input has shape [1, H, W] (single channel), expand to [1, 3, H, W]
        if x.size(1) == 1:
            x = x.expand(-1, 3, -1, -1)  # Safer than repeat for this purpose
        return self.model(x)
class ValidationTransform:
    def __init__(self, img_size=224):
        self.img_size = img_size

    def __call__(self, x):
        # Get image and label
        image, label = x['image'], x['label']
        
        # Convert to torch tensor first (from numpy) if not already
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image.astype(np.float32))
        else:
            image_tensor = image
        
        # Ensure correct dimensions and format
        if image_tensor.dim() == 2:  # [H, W]
            image_tensor = image_tensor.unsqueeze(0)  # Add channel dim [1, H, W]
        
        # Resize if needed
        if image_tensor.shape[1] != self.img_size or image_tensor.shape[2] != self.img_size:
            image_tensor = TF.resize(image_tensor, [self.img_size, self.img_size])
        
        # Process label - IMPORTANT: Convert to Long for CrossEntropyLoss
        if isinstance(label, np.ndarray):
            # Convert to Long type for class indices
            label_tensor = torch.from_numpy(label.astype(np.int64))
            
            # Resize label if needed
            if label_tensor.shape[0] != self.img_size or label_tensor.shape[1] != self.img_size:
                # Add channel dimension for resize operation if needed
                if label_tensor.dim() == 2:
                    label_tensor = label_tensor.unsqueeze(0)
                    
                label_tensor = TF.resize(label_tensor, [self.img_size, self.img_size], 
                                        interpolation=TF.InterpolationMode.NEAREST)
                
                # Remove channel dimension if it was added
                if label.ndim == 2:
                    label_tensor = label_tensor.squeeze(0)
        else:
            # If it's already a tensor, convert to Long
            label_tensor = label.long()
            
        return {
            'image': image_tensor,  # Will be [1, H, W] - single channel
            'label': label_tensor   # Will be [H, W] as Long type
        }
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. base lr = target lr / multiplier if multiplier < 1.0
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier > 1.0:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
        else:
            return [base_lr * (1. - (1. - self.multiplier) * (self.total_epoch - self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics)
            else:
                self.after_scheduler.step(metrics, epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != torch.optim.lr_scheduler.ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                return

            if epoch is None:
                epoch = self.last_epoch + 1
            self.last_epoch = epoch
            if self.last_epoch <= self.total_epoch:
                warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
                for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                    param_group['lr'] = lr
            else:
                if self.after_scheduler:
                    self.after_scheduler.step(epoch - self.total_epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

# Function to calculate per-class metrics
def calculate_per_class_metrics(outputs, labels, num_classes=2):
    """
    Calculate accuracy and CE loss for each class
    
    Args:
        outputs: model output logits [B, C, H, W]
        labels: ground truth labels [B, H, W]
        num_classes: number of classes (default: 2 for binary segmentation)
        
    Returns:
        dict: Dictionary containing per-class metrics
    """
    batch_size = outputs.size(0)
    
    # For CE loss calculation
    ce_loss = CrossEntropyLoss(reduction='none')
    pixel_losses = ce_loss(outputs, labels)
    
    # Convert to probabilities and predictions
    outputs_soft = torch.softmax(outputs, dim=1)
    outputs_pred = torch.argmax(outputs_soft, dim=1)
    
    # Initialize metrics
    metrics = {
        'accuracy_bg': 0.0,
        'accuracy_pa': 0.0,
        'ce_loss_bg': 0.0,
        'ce_loss_pa': 0.0,
        'pixel_count_bg': 0.0,
        'pixel_count_pa': 0.0
    }
    
    # Calculate metrics for each image in batch
    for b in range(batch_size):
        pred = outputs_pred[b]
        target = labels[b]
        loss_map = pixel_losses[b]
        
        # Create masks for each class
        mask_bg = (target == 0)
        mask_pa = (target == 1)
        
        # Count pixels for each class
        bg_pixels = mask_bg.sum().float()
        pa_pixels = mask_pa.sum().float()
        
        metrics['pixel_count_bg'] += bg_pixels.item()
        metrics['pixel_count_pa'] += pa_pixels.item()
        
        # Calculate accuracy for each class (correctly classified pixels / total pixels)
        if bg_pixels > 0:
            accuracy_bg = ((pred == 0) & mask_bg).sum().float() / bg_pixels
            metrics['accuracy_bg'] += accuracy_bg.item() * bg_pixels.item()
            
            # CE loss for background pixels
            ce_loss_bg = (loss_map * mask_bg.float()).sum() / bg_pixels
            metrics['ce_loss_bg'] += ce_loss_bg.item() * bg_pixels.item()
        
        if pa_pixels > 0:
            accuracy_pa = ((pred == 1) & mask_pa).sum().float() / pa_pixels
            metrics['accuracy_pa'] += accuracy_pa.item() * pa_pixels.item()
            
            # CE loss for PA pixels
            ce_loss_pa = (loss_map * mask_pa.float()).sum() / pa_pixels
            metrics['ce_loss_pa'] += ce_loss_pa.item() * pa_pixels.item()
    
    # Normalize metrics by pixel count
    total_bg_pixels = metrics['pixel_count_bg']
    total_pa_pixels = metrics['pixel_count_pa']
    
    if total_bg_pixels > 0:
        metrics['accuracy_bg'] /= total_bg_pixels
        metrics['ce_loss_bg'] /= total_bg_pixels
    
    if total_pa_pixels > 0:
        metrics['accuracy_pa'] /= total_pa_pixels
        metrics['ce_loss_pa'] /= total_pa_pixels
    
    return metrics

def trainer_parse_improved(args, model, snapshot_path):
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Initialize wandb
    wandb.init(project="transunet-pa-segmentation-improved", 
            name=f"parse_{args.vit_name}_improved_lr{args.base_lr}_bs{args.batch_size}",
            config={
                "learning_rate": args.base_lr,
                "batch_size": args.batch_size,
                "model": args.vit_name,
                "epochs": args.max_epochs,
                "img_size": args.img_size,
                "num_classes": args.num_classes,
                "pa_slice_ratio": args.pa_slice_ratio if hasattr(args, 'pa_slice_ratio') else 0.8,
                "patch_size": args.patch_size if hasattr(args, 'patch_size') else None,
                "optimizer": "AdamW",
                "scheduler": "Warmup+Cosine",
                "loss": "CE(0.3)+Tversky(0.7,beta=0.7)"
            })
    
    # Create logs directory if it doesn't exist
    os.makedirs('./logs', exist_ok=True)
    
    # Log filename based on experiment parameters
    log_filename = f'./logs/improved_{args.dataset}_{args.vit_name}_epo{args.max_epochs}_bs{args.batch_size}_lr{args.base_lr}.txt'
    
    # Setup logging to file
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.info(str(args))
    logging.info(f"Log file saved at: {os.path.abspath(log_filename)}")
    
    # Calculate class weights based on PA vs background ratio
    # PA is ~0.63% of voxels, background is ~99.37%
    bg_weight = 1.0
    pa_weight = 25.0
    class_weights = torch.tensor([bg_weight, pa_weight], dtype=torch.float32).cuda()
    
    # Wrap the model with our ModelWrapper before loading checkpoint
    model = ModelWrapper(model)
    
    # Try to load checkpoint
    model, start_epoch, best_performance = try_load_checkpoint(args, model)
    
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    # Create enhanced dataset with PA focus
    pa_slice_ratio = getattr(args, 'pa_slice_ratio', 0.8)
    patch_size = getattr(args, 'patch_size', None)
    if patch_size is not None and isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    
    # Use our enhanced dataset and augmentations
    db_train = PAFocusedDataset(
        base_dir=args.root_path, 
        list_dir=args.list_dir, 
        split="train",
        transform=EnhancedRandomGenerator(output_size=[args.img_size, args.img_size]),
        pa_threshold=0.001,
        pa_slice_ratio=pa_slice_ratio
    )
    
    db_val = PAFocusedDataset(
        base_dir=args.root_path, 
        list_dir=args.list_dir, 
        split="val",
        transform=ValidationTransform(img_size=args.img_size)
    )
    
    logging.info(f"The length of train set is: {len(db_train)}")
    logging.info(f"The length of validation set is: {len(db_val)}")

    # Use num_workers=0 for validation to avoid multiprocessing issues
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    tta = TestTimeAugmentation(num_augmentations=4)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
        
    model.train()
    
    # Use our improved loss function
    dice_loss = DiceLoss(num_classes)
    combined_loss = CombinedLoss(
        num_classes=num_classes,
        class_weights=class_weights,
        ce_weight=0.2,
        tversky_weight=0.6,
        focal_weight=0.2,
        tversky_beta=0.8,
    )
    
    # Use AdamW optimizer instead of SGD
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    
    # Setup warmup + cosine annealing scheduler
    # Warmup for 5% of total iterations
    total_iterations = args.max_epochs * len(trainloader)
    warmup_iterations = int(0.05 * total_iterations)
    warmup_epochs = max(1, int(0.05 * args.max_epochs))
    
    # Use cosine annealing after warmup
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.max_epochs - warmup_epochs,
        eta_min=1e-6
    )
    
    # Combine with warmup scheduler
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1.0,
        total_epoch=warmup_epochs,
        after_scheduler=cosine_scheduler
    )
    
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations")
    
    best_epoch = start_epoch
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    
    # Create a table for sample predictions
    wandb_pred_table = wandb.Table(columns=["epoch", "iteration", "image", "prediction", "ground_truth", "dice"])
    
    # Define validation frequency
    val_frequency = args.val_frequency if hasattr(args, 'val_frequency') else 2000
    
    for epoch_num in iterator:
        epoch_loss = 0
        epoch_combined_loss = 0
        
        # Track per-class metrics for the epoch
        epoch_metrics = {
            'accuracy_bg': 0.0,
            'accuracy_pa': 0.0,
            'ce_loss_bg': 0.0,
            'ce_loss_pa': 0.0,
            'pixel_count_bg': 0,
            'pixel_count_pa': 0
        }
        
        step_count = 0
        
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            #Convert label to Long type
            label_batch = label_batch.long()
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            outputs = model(image_batch)
            
            # Apply combined loss
            loss = combined_loss(outputs, label_batch)
            
            # Calculate dice loss separately for tracking
            dice_value = dice_loss(outputs, label_batch, softmax=True)
            
            # Calculate per-class metrics
            batch_metrics = calculate_per_class_metrics(outputs, label_batch, num_classes)
            
            # Update epoch per-class metrics
            for key in epoch_metrics:
                if key.startswith('pixel_count'):
                    epoch_metrics[key] += batch_metrics[key]
                else:
                    epoch_metrics[key] += batch_metrics[key] * batch_metrics['pixel_count_bg' if 'bg' in key else 'pixel_count_pa']
            
            # Update epoch stats
            epoch_loss += dice_value.item()  # Track dice loss for comparison with baseline
            epoch_combined_loss += loss.item()
            step_count += 1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('info/combined_loss', loss, iter_num)
            writer.add_scalar('info/dice_loss', dice_value, iter_num)
            
            # Log per-class metrics to wandb
            wandb.log({
                "iteration": iter_num,
                "combined_loss": loss.item(),
                "dice_loss": dice_value.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "train_accuracy_background": batch_metrics['accuracy_bg'],
                "train_accuracy_pa": batch_metrics['accuracy_pa'],
                "train_ce_loss_background": batch_metrics['ce_loss_bg'],
                "train_ce_loss_pa": batch_metrics['ce_loss_pa'],
                "train_pixel_ratio_pa": batch_metrics['pixel_count_pa'] / (batch_metrics['pixel_count_bg'] + batch_metrics['pixel_count_pa'] + 1e-8)
            })

            logging.info(f'iteration {iter_num} : combined_loss : {loss.item():.4f}, dice_loss: {dice_value.item():.4f}, '
                        f'bg_acc: {batch_metrics["accuracy_bg"]:.4f}, pa_acc: {batch_metrics["accuracy_pa"]:.4f}')

            # Sample visualization
            if iter_num % 20 == 0 and image_batch.size(0) > 0:
                with torch.no_grad():
                    # Get a sample image
                    image = image_batch[0, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                    
                    # Get prediction
                    outputs_soft = torch.softmax(outputs, dim=1)
                    outputs_pred = torch.argmax(outputs_soft, dim=1, keepdim=True)
                    
                    # Log images
                    writer.add_image('train/Image', image, iter_num)
                    writer.add_image('train/Prediction', outputs_pred[0, ...] * 50, iter_num)
                    writer.add_image('train/GroundTruth', label_batch[0, ...].unsqueeze(0) * 50, iter_num)
                    
                    # Calculate Dice score for the sample
                    pred_np = outputs_pred[0, 0].cpu().numpy()
                    label_np = label_batch[0].cpu().numpy()
                    dice_sample = 2 * np.sum(pred_np * label_np) / (np.sum(pred_np) + np.sum(label_np) + 1e-5)
                    
                    # Log images to wandb
                    fig = plt.figure(figsize=(15, 5))
                    plt.subplot(131)
                    plt.title("Image")
                    plt.imshow(image[0].cpu().numpy(), cmap='gray')
                    plt.axis('off')
                    
                    plt.subplot(132)
                    plt.title(f"Prediction (Dice: {dice_sample:.4f})")
                    plt.imshow(pred_np, cmap='Blues')
                    plt.axis('off')
                    
                    plt.subplot(133)
                    plt.title("Ground Truth")
                    plt.imshow(label_np, cmap='Blues')
                    plt.axis('off')
                    
                    wandb.log({
                        "sample_visualization": wandb.Image(fig),
                        "sample_dice": dice_sample
                    })
                    plt.close(fig)
                    
                    # Add sample to table (periodically)
                    if iter_num % 50 == 0:
                        img_to_log = wandb.Image(image[0].cpu().numpy())
                        pred_to_log = wandb.Image(pred_np)
                        gt_to_log = wandb.Image(label_np)
                        wandb_pred_table.add_data(epoch_num, iter_num, img_to_log, pred_to_log, gt_to_log, dice_sample)
            
            # Run validation
            run_validation = (iter_num % val_frequency == 0) or (i_batch == len(trainloader) - 1)
            
            if run_validation:
                logging.info(f"Running validation at iteration {iter_num}")
                model.eval()
                val_loss = 0
                val_dice = 0
                val_steps = 0
                
                # Track validation per-class metrics
                val_metrics = {
                    'accuracy_bg': 0.0,
                    'accuracy_pa': 0.0,
                    'ce_loss_bg': 0.0,
                    'ce_loss_pa': 0.0,
                    'pixel_count_bg': 0,
                    'pixel_count_pa': 0
                }
                
                with torch.no_grad():
                    for val_batch in valloader:
                        val_images, val_labels = val_batch['image'], val_batch['label']
                        # Convert label to Long type
                        val_labels = val_labels.long()
                        val_images, val_labels = val_images.cuda(), val_labels.cuda()
                        
                        # Get model output for loss calculation
                        val_outputs = model(val_images)
                        val_combined_loss = combined_loss(val_outputs, val_labels)
                        val_dice_loss = dice_loss(val_outputs, val_labels, softmax=True)
                        
                        # Apply test-time augmentation for prediction metrics
                        val_outputs_soft = tta(model, val_images)
                        val_outputs_pred = torch.argmax(val_outputs_soft, dim=1)
                        
                        # Calculate per-class metrics for validation
                        batch_val_metrics = calculate_per_class_metrics(val_outputs, val_labels, num_classes)
                        
                        # Update validation metrics
                        for key in val_metrics:
                            if key.startswith('pixel_count'):
                                val_metrics[key] += batch_val_metrics[key]
                            else:
                                val_metrics[key] += batch_val_metrics[key] * batch_val_metrics['pixel_count_bg' if 'bg' in key else 'pixel_count_pa']
                        
                        val_loss += val_combined_loss.item()
                        
                        # Calculate Dice for validation batch using TTA predictions
                        for b in range(val_labels.shape[0]):
                            pred = val_outputs_pred[b].cpu().numpy()
                            target = val_labels[b].cpu().numpy()
                            dice = 2 * np.sum(pred * target) / (np.sum(pred) + np.sum(target) + 1e-5)
                            val_dice += dice
                        
                        val_steps += val_labels.shape[0]
                # Log validation images             
                # Normalize validation metrics by pixel count
                for key in val_metrics:
                    if not key.startswith('pixel_count'):
                        pixel_count_key = 'pixel_count_bg' if 'bg' in key else 'pixel_count_pa'
                        if val_metrics[pixel_count_key] > 0:
                            val_metrics[key] /= val_metrics[pixel_count_key]
                
                avg_val_loss = val_loss / max(val_steps, 1)  # Avoid division by zero
                avg_val_dice = val_dice / max(val_steps, 1)
                
                logging.info(f"Validation at iteration {iter_num}: Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}, "
                            f"BG Acc: {val_metrics['accuracy_bg']:.4f}, PA Acc: {val_metrics['accuracy_pa']:.4f}")
                
                # Log validation metrics to wandb
                wandb.log({
                    "val_loss": avg_val_loss,
                    "val_dice": avg_val_dice,
                    "val_iteration": iter_num,
                    "val_accuracy_background": val_metrics['accuracy_bg'],
                    "val_accuracy_pa": val_metrics['accuracy_pa'],
                    "val_ce_loss_background": val_metrics['ce_loss_bg'],
                    "val_ce_loss_pa": val_metrics['ce_loss_pa'],
                    "val_pixel_ratio_pa": val_metrics['pixel_count_pa'] / (val_metrics['pixel_count_bg'] + val_metrics['pixel_count_pa'] + 1e-8)
                })
                
                writer.add_scalar('val/loss', avg_val_loss, iter_num)
                writer.add_scalar('val/dice', avg_val_dice, iter_num)
                writer.add_scalar('val/accuracy_bg', val_metrics['accuracy_bg'], iter_num)
                writer.add_scalar('val/accuracy_pa', val_metrics['accuracy_pa'], iter_num)
                writer.add_scalar('val/ce_loss_bg', val_metrics['ce_loss_bg'], iter_num)
                writer.add_scalar('val/ce_loss_pa', val_metrics['ce_loss_pa'], iter_num)
                
                # Create and log class metrics chart 
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Accuracy plot
                ax1.bar(['Background', 'PA'], [val_metrics['accuracy_bg'], val_metrics['accuracy_pa']], color=['blue', 'orange'])
                ax1.set_title(f'Validation Accuracy by Class (Iter {iter_num})')
                ax1.set_ylim(0, 1.0)
                ax1.set_ylabel('Accuracy')
                
                # CE Loss plot
                ax2.bar(['Background', 'PA'], [val_metrics['ce_loss_bg'], val_metrics['ce_loss_pa']], color=['blue', 'orange'])
                ax2.set_title(f'Validation CE Loss by Class (Iter {iter_num})')
                ax2.set_ylabel('Cross Entropy Loss')
                
                plt.tight_layout()
                wandb.log({"val_class_metrics": wandb.Image(fig)})
                plt.close(fig)
                
                # Save best model
                is_best = avg_val_dice > best_performance
                if is_best:
                    best_performance = avg_val_dice
                    best_epoch = epoch_num
                    
                    # Save to original snapshot path
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    if isinstance(model, nn.DataParallel):
                        torch.save(model.module.state_dict(), save_mode_path)
                    else:
                        torch.save(model.state_dict(), save_mode_path)
                    
                    # Also save checkpoint
                    save_checkpoint(model, epoch_num, best_performance, args, is_best=True)
                    
                    logging.info(f"Saved new best model with validation Dice: {best_performance:.4f}")
                
                # Switch back to training mode
                model.train()
        
        # Update scheduler at the end of epoch
        scheduler.step()

        # Normalize epoch metrics by pixel count
        for key in epoch_metrics:
            if not key.startswith('pixel_count'):
                pixel_count_key = 'pixel_count_bg' if 'bg' in key else 'pixel_count_pa'
                if epoch_metrics[pixel_count_key] > 0:
                    epoch_metrics[key] /= epoch_metrics[pixel_count_key]

        # Log epoch stats
        avg_loss = epoch_loss / max(step_count, 1)  # Avoid division by zero
        avg_combined_loss = epoch_combined_loss / max(step_count, 1)
        
        wandb.log({
            "epoch": epoch_num,
            "epoch_dice_loss": avg_loss,
            "epoch_combined_loss": avg_combined_loss,
            "epoch_accuracy_background": epoch_metrics['accuracy_bg'],
            "epoch_accuracy_pa": epoch_metrics['accuracy_pa'],
            "epoch_ce_loss_background": epoch_metrics['ce_loss_bg'],
            "epoch_ce_loss_pa": epoch_metrics['ce_loss_pa']
        })
        
        # Create and log epoch class metrics chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy plot for epoch
        ax1.bar(['Background', 'PA'], [epoch_metrics['accuracy_bg'], epoch_metrics['accuracy_pa']], color=['blue', 'orange'])
        ax1.set_title(f'Epoch {epoch_num} Accuracy by Class')
        ax1.set_ylim(0, 1.0)
        ax1.set_ylabel('Accuracy')
        
        # CE Loss plot for epoch
        ax2.bar(['Background', 'PA'], [epoch_metrics['ce_loss_bg'], epoch_metrics['ce_loss_pa']], color=['blue', 'orange'])
        ax2.set_title(f'Epoch {epoch_num} CE Loss by Class')
        ax2.set_ylabel('Cross Entropy Loss')
        
        plt.tight_layout()
        wandb.log({"epoch_class_metrics": wandb.Image(fig)})
        plt.close(fig)
        
        logging.info(f'Epoch {epoch_num} : average dice loss : {avg_loss:.4f}, combined loss: {avg_combined_loss:.4f}, '
                    f'bg_acc: {epoch_metrics["accuracy_bg"]:.4f}, pa_acc: {epoch_metrics["accuracy_pa"]:.4f}')

        # Save model periodically
        save_interval = 5
        if (epoch_num + 1) % save_interval == 0 or epoch_num == max_epoch - 1:
            # Save to original snapshot path
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_mode_path)
            else:
                torch.save(model.state_dict(), save_mode_path)
            
            # Also save checkpoint
            save_checkpoint(model, epoch_num, best_performance, args)
            
            logging.info(f"Saved model to {save_mode_path}")

    # Upload prediction samples table
    wandb.log({"prediction_samples": wandb_pred_table})
    
    # Log final best performance
    logging.info(f"Best validation performance: {best_performance:.4f} at epoch {best_epoch}")
    wandb.log({"best_val_dice": best_performance, "best_epoch": best_epoch})
    
    # Close wandb
    wandb.finish()
    writer.close()
    return "Training Finished!"

# Helper functions
def try_load_checkpoint(args, model):
    """
    Try to load a checkpoint if it exists
    """
    # Create checkpoints directory if it doesn't exist
    os.makedirs('./checkpoints_enhanced_Focal_attention', exist_ok=True)
    
    # Define checkpoint path for this model
    checkpoint_path = f'./checkpoints_enhanced_Focal_attention/best_model_{args.dataset}_{args.vit_name}.pth'
    
    start_epoch = 0
    best_performance = 0.0
    
    # Check if resume flag is set
    if not args.resume:
        logging.info("Resume flag not set. Starting from scratch.")
        return model, start_epoch, best_performance
    
    # Check if checkpoint exists
    if os.path.exists(checkpoint_path):
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # If using ModelWrapper, we need to load into model.model
                if isinstance(model, ModelWrapper):
                    # Try to load into the wrapped model
                    try:
                        model.model.load_state_dict(checkpoint['model_state_dict'])
                    except Exception as e:
                        logging.warning(f"Failed to load directly into wrapped model: {str(e)}. Trying alternate approach.")
                        # If failed, try to load with the wrapper prefix
                        state_dict = {}
                        for k, v in checkpoint['model_state_dict'].items():
                            if k.startswith('model.'):
                                state_dict[k] = v
                            else:
                                state_dict[f'model.{k}'] = v
                        model.load_state_dict(state_dict)
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
                
                start_epoch = checkpoint['epoch'] + 1
                best_performance = checkpoint['best_performance']
                logging.info(f"Resuming from epoch {start_epoch} with best performance {best_performance:.4f}")
            else:
                # Just load state dict
                if isinstance(model, ModelWrapper):
                    # Try to load into the wrapped model
                    try:
                        model.model.load_state_dict(checkpoint)
                    except:
                        # Try to update keys with 'model.' prefix
                        state_dict = {}
                        for k, v in checkpoint.items():
                            state_dict[f'model.{k}'] = v
                        model.load_state_dict(state_dict)
                else:
                    model.load_state_dict(checkpoint)
                
                logging.info(f"Loaded model weights from {checkpoint_path}")
            
            return model, start_epoch, best_performance
            
        except Exception as e:
            logging.warning(f"Failed to load checkpoint: {str(e)}. Starting from scratch.")
    
    logging.info("No checkpoint found. Starting from scratch.")
    return model, start_epoch, best_performance

def save_checkpoint(model, epoch, best_performance, args, is_best=False):
    """
    Save model checkpoint
    """
    # Create checkpoints directory if it doesn't exist
    os.makedirs('./checkpoints_enhanced_Focal_attention', exist_ok=True)
    
    # Define checkpoint and best model paths
    checkpoint_path = f'./checkpoints_enhanced_Focal_attention/model_{args.dataset}_{args.vit_name}_epoch{epoch}.pth'
    best_model_path = f'./checkpoints_enhanced_Focal_attention/best_model_{args.dataset}_{args.vit_name}.pth'
    
    # Create checkpoint dict
    if isinstance(model, nn.DataParallel):
        if isinstance(model.module, ModelWrapper):
            state_dict = model.module.model.state_dict()  # Get the inner model state dict
        else:
            state_dict = model.module.state_dict()
    else:
        if isinstance(model, ModelWrapper):
            state_dict = model.model.state_dict()  # Get the inner model state dict
        else:
            state_dict = model.state_dict()
        
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'best_performance': best_performance
    }
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save best model if this is the best performance
    if is_best:
        torch.save(checkpoint, best_model_path)
        logging.info(f"Saved best model to {best_model_path} with performance {best_performance:.4f}")