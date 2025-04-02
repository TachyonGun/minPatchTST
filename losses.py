import torch
import torch.nn as nn

class MaskedMSELoss(nn.Module):
    """Loss function for masked pretraining as used in PatchTST paper"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)
        
    def forward(self, y_pred, y_true, mask=None):
        """
        Args:
            y_pred: Predicted values
            y_true: Target values  
            mask: Optional mask tensor (1 for masked values, 0 for unmasked)
        """
        if mask is None:
            return self.mse(y_pred, y_true)
        
        # Apply mask
        mask = mask.bool()
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        return self.mse(y_pred, y_true)

class ForecastingLoss(nn.Module):
    """Standard MSE loss for forecasting task"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        
    def forward(self, y_pred, y_true):
        return self.mse(y_pred, y_true) 