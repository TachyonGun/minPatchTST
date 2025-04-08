#from PatchTST.PatchTST_self_supervised.src.models.patchTST import PatchTST
from patchtst.patchTST import PatchTST
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

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


import matplotlib.pyplot as plt
import numpy as np

def plot_signal(patches, ground_truth=None, reconstruction=None, mask=None, sample_idx=0, num_channels=3):
    num_patches, channels, patch_size = patches.shape[1], patches.shape[2], patches.shape[3]
    total_time = num_patches * (patch_size // 2) + (patch_size // 2)

    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 1.5*num_channels), sharex=True)

    def reconstruct_full_signal(data, apply_mask=False):
        full_signal = np.zeros((channels, total_time))
        for p in range(num_patches):
            start = p * (patch_size // 2)
            end = start + patch_size
            if not apply_mask or (mask is None or mask[sample_idx, p]):
                segment = data[sample_idx, p, :, :]
                if hasattr(segment, 'detach'):
                    segment = segment.detach().cpu().numpy()
                full_signal[:, start:end] = segment
        return full_signal

    full_signal = reconstruct_full_signal(patches)

    gt_signal = np.full((channels, total_time), np.nan)
    recon_signal = np.full((channels, total_time), np.nan)

    if ground_truth is not None and reconstruction is not None and mask is not None:
        for p in range(num_patches):
            if mask[sample_idx, p]:
                start = p * (patch_size // 2)
                end = start + patch_size
                gt_segment = ground_truth[sample_idx, p]
                recon_segment = reconstruction[sample_idx, p]

                if hasattr(gt_segment, 'detach'):
                    gt_segment = gt_segment.detach().cpu().numpy()
                if hasattr(recon_segment, 'detach'):
                    recon_segment = recon_segment.detach().cpu().numpy()

                gt_signal[:, start:end] = gt_segment
                recon_signal[:, start:end] = recon_segment

    for i in range(num_channels):
        axes[i].plot(np.arange(total_time), full_signal[i, :], linewidth=0.8, label='Original', color='black')
        if ground_truth is not None:
            axes[i].plot(np.arange(total_time), gt_signal[i, :], 'b:', linewidth=0.8, label='Ground Truth')
        if reconstruction is not None:
            axes[i].plot(np.arange(total_time), recon_signal[i, :], 'r:', linewidth=0.8, label='Reconstruction')

        axes[i].set_ylabel(f'Ch {i+1}')
        axes[i].set_yticks([])
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
        axes[i].margins(y=0.1)

        for t in range(0, total_time + 1, 125):
            axes[i].axvline(x=t, color='gray', linestyle='--', linewidth=0.6)

    axes[-1].set_xlabel('Time Steps')
    axes[-1].set_xlim(0, total_time - 1)

    axes[0].legend(loc='upper right')

    print(f"Plotting first {num_channels} channels ...")
    plt.tight_layout()
    plt.show()
