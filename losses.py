import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), f"Input sizes {inputs.size()} and target sizes {target.size()} must match"
        class_wise_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class TverskyLoss(nn.Module):
    def __init__(self, n_classes, alpha=0.3, beta=0.7, smooth=1e-5):
        """
        Tversky loss for imbalanced data
        :param alpha: weight of false positives
        :param beta: weight of false negatives (higher beta prioritizes recall)
        """
        super(TverskyLoss, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        
        target = self._one_hot_encoder(target)
        
        if weight is None:
            weight = [1] * self.n_classes
            
        assert inputs.size() == target.size(), f"Input sizes {inputs.size()} and target sizes {target.size()} must match"
        
        loss = 0.0
        
        for i in range(self.n_classes):
            # Calculate Tversky for each class
            tp = torch.sum(inputs[:, i] * target[:, i])
            fp = torch.sum(inputs[:, i] * (1 - target[:, i]))
            fn = torch.sum((1 - inputs[:, i]) * target[:, i])
            
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            loss += (1 - tversky) * weight[i]
            
        return loss / self.n_classes

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p_t = torch.exp(-ce_loss)
        loss = (1 - p_t) ** self.gamma * ce_loss
        return loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None, ce_weight=0.3, tversky_weight=0.7, 
                 focal_weight=0.3, tversky_beta=0.7):
        """
        Combined loss function for medical image segmentation with imbalanced classes
        :param num_classes: Number of classes
        :param class_weights: Weight for each class in CE loss
        :param ce_weight: Weight for CE loss component
        :param tversky_weight: Weight for Tversky loss component
        :param focal_weight: Weight for Focal loss component
        :param tversky_beta: Beta parameter for Tversky loss (higher values prioritize recall)
        """
        super(CombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
        
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.tversky = TverskyLoss(num_classes, alpha=1-tversky_beta, beta=tversky_beta)
        # Use existing FocalLoss with gamma=2 for hard example focus
        self.focal = FocalLoss(gamma=2, alpha=class_weights)
    
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        tversky_loss = self.tversky(inputs, targets)
        focal_loss = self.focal(inputs, targets)
        
        return (self.ce_weight * ce_loss) + (self.tversky_weight * tversky_loss) + (self.focal_weight * focal_loss)