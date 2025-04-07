#from PatchTST.PatchTST_self_supervised.src.models.patchTST import PatchTST
from patchtst.patchTST import PatchTST


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