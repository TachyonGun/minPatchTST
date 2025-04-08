#from PatchTST.PatchTST_self_supervised.src.models.patchTST import PatchTST
from patchtst.patchTST import PatchTST
import torch
import torch.nn as nn

def create_patchtst_model(c_in, target_dim, patch_len, stride, num_patch,
                n_layers=3, d_model=128, n_heads=16, shared_embedding=True,
                d_ff=256, dropout=0.2, head_dropout=0.2, head_type="pretrain"):
    """
    Create PatchTST model with specified parameters
    
    Args:
        c_in: Number of input variables/features
        target_dim: Prediction horizon length
        patch_len: Length of each patch
        stride: Stride between patches
        num_patch: Number of patches
        n_layers: Number of transformer layers
        d_model: Model dimension
        n_heads: Number of attention heads
        shared_embedding: Whether to share embeddings across variables
        d_ff: Dimension of feedforward network
        dropout: Dropout rate
        head_dropout: Head dropout rate
        head_type: Type of head to use ("pretrain", "prediction", "regression", "classification")
    """
    model = PatchTST(
        c_in=c_in,
        target_dim=target_dim,
        patch_len=patch_len,
        stride=stride,
        num_patch=num_patch,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        shared_embedding=shared_embedding,
        d_ff=d_ff,
        dropout=dropout,
        head_dropout=head_dropout,
        head_type=head_type,
        norm='LayerNorm'
    )
    
    return model

def get_num_patches(context_points, patch_len, stride):
    """Calculate number of patches given sequence length and patch parameters"""
    return (max(context_points, patch_len) - patch_len) // stride + 1 



class RevIN(nn.Module):
    """
    Reversible Instance Normalization from https://github.com/ts-kim/RevIN
    with minor modifications to match PatchTST implementation
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: 
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        # x shape: [batch_size, num_patches, n_vars, patch_len]
        # Compute statistics over patches and patch_len dimensions
        dim2reduce = (1, 3)  # Reduce over num_patches and patch_len
        if self.subtract_last:
            self.last = x[:,-1,:,:].unsqueeze(1)  # Keep last patch
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        # x shape: [batch_size, num_patches, n_vars, patch_len]
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            # Reshape affine parameters for proper broadcasting
            x = x * self.affine_weight.view(1, 1, -1, 1)
            x = x + self.affine_bias.view(1, 1, -1, 1)
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias.view(1, 1, -1, 1)
            x = x / (self.affine_weight.view(1, 1, -1, 1) + self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
