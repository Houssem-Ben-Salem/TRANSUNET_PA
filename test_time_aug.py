import torch
import torch.nn.functional as F

class TestTimeAugmentation:
    def __init__(self, num_augmentations=4):
        """
        Test-time augmentation for more robust inference
        
        Args:
            num_augmentations: Number of augmentations to perform (1-4)
        """
        self.num_augmentations = min(num_augmentations, 4)  # Limit to max 4 augmentations
        
    def __call__(self, model, image):
        """
        Apply test-time augmentation
        
        Args:
            model: The model to use for predictions
            image: Input image tensor [B, C, H, W]
            
        Returns:
            Averaged prediction after TTA
        """
        model.eval()
        predictions = []
        
        # Original prediction
        with torch.no_grad():
            orig_pred = model(image)
            orig_soft = F.softmax(orig_pred, dim=1)
            predictions.append(orig_soft)
        
        # Only apply augmentations if requested
        if self.num_augmentations >= 1:
            # Horizontal flip
            with torch.no_grad():
                flipped = torch.flip(image, dims=[3])
                pred = model(flipped)
                pred_soft = F.softmax(pred, dim=1)
                # Flip back the prediction
                pred_soft = torch.flip(pred_soft, dims=[3])
                predictions.append(pred_soft)
        
        if self.num_augmentations >= 2:
            # Vertical flip
            with torch.no_grad():
                flipped = torch.flip(image, dims=[2])
                pred = model(flipped)
                pred_soft = F.softmax(pred, dim=1)
                # Flip back the prediction
                pred_soft = torch.flip(pred_soft, dims=[2])
                predictions.append(pred_soft)
        
        if self.num_augmentations >= 3:
            # 90 degree rotation
            with torch.no_grad():
                rotated = torch.rot90(image, k=1, dims=[2, 3])
                pred = model(rotated)
                pred_soft = F.softmax(pred, dim=1)
                # Rotate back the prediction (270 degrees)
                pred_soft = torch.rot90(pred_soft, k=3, dims=[2, 3])
                predictions.append(pred_soft)
        
        if self.num_augmentations >= 4:
            # 180 degree rotation
            with torch.no_grad():
                rotated = torch.rot90(image, k=2, dims=[2, 3])
                pred = model(rotated)
                pred_soft = F.softmax(pred, dim=1)
                # Rotate back the prediction (180 degrees)
                pred_soft = torch.rot90(pred_soft, k=2, dims=[2, 3])
                predictions.append(pred_soft)
        
        # Average predictions
        avg_pred = torch.stack(predictions).mean(dim=0)
        return avg_pred