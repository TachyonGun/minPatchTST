import wandb
import torch
import matplotlib.pyplot as plt
import numpy as np
from models import create_patchtst_model, get_num_patches
import os
import math
from pretrain import RevIN
from pretrain import plot_reconstruction, get_fixed_mask, get_fixed_sample, create_dataloader


#wandb.init()
DATASET = 'seed_iv/session'
USE_WANDB = False

#Custom dataset configuration (if using GenericArrayDataset)
WEIGHT_DECAY = 0.0 # 1e-3  # Weight decay for regularization
USE_GENERIC_DATASET = True # if 'all_six_datasets' not in DATASET else False # Set to True to use GenericArrayDataset
GENERIC_CONFIG = {
    'context_points': 2000,    # Number of input timesteps
    'target_points': 0,      # Number of timesteps to predict if doing forecasting
    'patch_len': 200,          # Length of each patch
    'stride': 100,              # Stride between patches (if it equals patch_len, no overlap)
    'batch_size': int(64),          # Batch size for training
    'mask_ratio': 0.3,         # Ratio of patches to mask
    'n_epochs': 20,            # Number of training epochs
    'd_model': 768 // 2,           # Model dimension
    'n_heads': 6,            # Number of attention heads
    'd_ff': 768 * 2,             # Feed-forward dimension
    'dropout': 0.05,          # Dropout rate
    'head_dropout': 0.01,     # Head dropout rate
    'use_revin': True,       # Whether to use RevIN
    'revin_affine': False,    # RevIN affine parameter
    'revin_eps': 1e-5,       # RevIN epsilon
    'subtract_last': False,    # RevIN subtract last
    'weight_decay': WEIGHT_DECAY     # Weight decay for regularization
}

# Model hyperparameters (matching paper configuration)
if USE_GENERIC_DATASET:
    # Use configuration from GENERIC_CONFIG
    CONTEXT_POINTS = GENERIC_CONFIG['context_points']
    TARGET_POINTS = GENERIC_CONFIG['target_points']
    PATCH_LEN = GENERIC_CONFIG['patch_len']
    STRIDE = GENERIC_CONFIG['stride']
    BATCH_SIZE = GENERIC_CONFIG['batch_size']
    MASK_RATIO = GENERIC_CONFIG['mask_ratio']
    N_EPOCHS = GENERIC_CONFIG['n_epochs']
    D_MODEL = GENERIC_CONFIG['d_model']
    N_HEADS = GENERIC_CONFIG['n_heads']
    D_FF = GENERIC_CONFIG['d_ff']
    DROPOUT = GENERIC_CONFIG['dropout']
    HEAD_DROPOUT = GENERIC_CONFIG['head_dropout']
    USE_REVIN = GENERIC_CONFIG['use_revin']
    REVIN_AFFINE = GENERIC_CONFIG['revin_affine']
    REVIN_EPS = GENERIC_CONFIG['revin_eps']
    SUBTRACT_LAST = GENERIC_CONFIG['subtract_last']
    WEIGHT_DECAY = GENERIC_CONFIG['weight_decay']

else:
    # Original hyperparameters for other datasets
    CONTEXT_POINTS = 512
    TARGET_POINTS = 96
    PATCH_LEN = 12
    STRIDE = 12
    BATCH_SIZE = 64
    MASK_RATIO = 0.8
    N_EPOCHS = 10
    D_MODEL = 128
    N_HEADS = 16
    D_FF = 512
    DROPOUT = 0.2
    HEAD_DROPOUT = 0.2
    USE_REVIN = True
    REVIN_AFFINE = True
    REVIN_EPS = 1e-5
    SUBTRACT_LAST = False


#############################################################
# Load data
#############################################################

# get labels 
train_loader, val_loader, _, column_names = create_dataloader(
    DATASET,
    CONTEXT_POINTS,
    TARGET_POINTS,
    PATCH_LEN,
    STRIDE,
    batch_size=BATCH_SIZE
)

fixed_train_sample = get_fixed_sample(train_loader)

# Create fixed masks if needed
fixed_train_mask = get_fixed_mask(fixed_train_sample.size(0), 
                                get_num_patches(CONTEXT_POINTS, PATCH_LEN, STRIDE), 
                                MASK_RATIO).cuda()


n_vars = fixed_train_sample.shape[2]
num_patches = get_num_patches(CONTEXT_POINTS, PATCH_LEN, STRIDE)
DEVICE = 'cuda'

revin = RevIN(n_vars, eps=REVIN_EPS, affine=REVIN_AFFINE, subtract_last=SUBTRACT_LAST) if USE_REVIN else None
if revin is not None:
    revin = revin.to('cuda')


#############################################################
# Create model
#############################################################

model = create_patchtst_model(
    c_in=n_vars,
    target_dim=TARGET_POINTS,
    patch_len=PATCH_LEN,
    stride=STRIDE,
    num_patch=num_patches,
    n_layers=6,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    d_ff=D_FF,
    dropout=DROPOUT,
    head_dropout=HEAD_DROPOUT
).to(DEVICE)

model_name = DATASET.replace('/', '_')
model_path = f'checkpoints/pretrained_{model_name}.pth'
checkpoint = torch.load(model_path, map_location='cpu')

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.to('cuda')
model.eval()
print('Done loading model!')


#############################################################
# Visualize attention
#############################################################

# Initialize attention cache
ATTENTION_CACHE = None

# Define attention hook functions
def read_attention_hook_scores(module, input, output):

    
    Q = input[0]
    K = input[1]
    V = input[2]

    bs = Q.size(0)

    # Linear (+ split in multiple heads)
    q_s = module.W_Q(Q).view(bs, -1, module.n_heads, module.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
    k_s = module.W_K(K).view(bs, -1, module.n_heads, module.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
    v_s = module.W_V(V).view(bs, -1, module.n_heads, module.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

    # Apply Scaled Dot-Product Attention (multiple heads)
    if module.res_attention:
        output, attn_weights, attn_scores = module.sdp_attn(q_s, k_s, v_s, prev=None, key_padding_mask=None, attn_mask=None)
    else:
        output, attn_weights = module.sdp_attn(q_s, k_s, v_s, key_padding_mask=None, attn_mask=None)
    # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]
    global ATTENTION_CACHE
    ATTENTION_CACHE = attn_scores
    
    # back to the original inputs dimensions
    output = output.transpose(1, 2).contiguous().view(bs, -1, module.n_heads * module.d_v) # output: [bs x q_len x n_heads * d_v]
    output = module.to_out(output)

    if module.res_attention: return output, attn_weights, attn_scores
    else: return output, attn_weights

    # Return output with uniform attention
    return new_attn_output, uniform_attention, uniform_attention  # Return uniform_attention as scores too

def read_attention_hook_weights(module, input, output):

    
    Q = input[0]
    K = input[1]
    V = input[2]

    bs = Q.size(0)

    # Linear (+ split in multiple heads)
    q_s = module.W_Q(Q).view(bs, -1, module.n_heads, module.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
    k_s = module.W_K(K).view(bs, -1, module.n_heads, module.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
    v_s = module.W_V(V).view(bs, -1, module.n_heads, module.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

    # Apply Scaled Dot-Product Attention (multiple heads)
    if module.res_attention:
        output, attn_weights, attn_scores = module.sdp_attn(q_s, k_s, v_s, prev=None, key_padding_mask=None, attn_mask=None)
    else:
        output, attn_weights = module.sdp_attn(q_s, k_s, v_s, key_padding_mask=None, attn_mask=None)
    # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]
    global ATTENTION_CACHE
    ATTENTION_CACHE = attn_weights
    
    # back to the original inputs dimensions
    output = output.transpose(1, 2).contiguous().view(bs, -1, module.n_heads * module.d_v) # output: [bs x q_len x n_heads * d_v]
    output = module.to_out(output)

    if module.res_attention: return output, attn_weights, attn_scores
    else: return output, attn_weights

    # Return output with uniform attention
    return new_attn_output, uniform_attention, uniform_attention  # Return uniform_attention as scores too

def visualize_attention_with_signal_head_average_channel_wise(fixed_train_sample, attention_per_channel, sample_idx=0, patch_len=100, stride=100, save=False, save_dir=''):
    """
    Visualize attention patterns and signals for all channels in a grid layout.
    
    Args:
        fixed_train_sample: tensor of shape [n_samples, n_patches, n_channels, patch_len]
        attention_per_channel: tensor of shape [n_samples, n_channels, n_heads, n_patches, n_patches]
        sample_idx: which sample to visualize
        patch_len: length of each patch
        stride: stride between patches
    """
    n_channels = fixed_train_sample.shape[2]
    n_patches = fixed_train_sample.shape[1]
    n_heads = attention_per_channel.shape[2]
    
    # Calculate layout
    n_cols = min(12, n_channels) # was 6
    n_rows = math.ceil(n_channels / n_cols) * 2  # Multiply by 2 for attention + signal plots
    
    # Create figure
    fig = plt.figure(figsize=(5*n_cols, 4*n_rows))
    
    # For each channel
    for channel_idx in range(n_channels):
        # Calculate row and column position
        row = (channel_idx // n_cols) * 2  # Multiply by 2 to leave space for signal plot
        col = channel_idx % n_cols
        
        # Get signal for this channel
        signal = fixed_train_sample[sample_idx, :, channel_idx, :].flatten()
        
        # Average attention across heads for this channel
        attention_matrix = attention_per_channel[sample_idx, channel_idx].mean(dim=0)  # Average over heads
        
        # Calculate attention received by each timestep
        attention_per_patch = attention_matrix.mean(dim=0)  # Average attention received by each patch
        
        # Calculate total sequence length
        seq_len = (n_patches - 1) * stride + patch_len
        
        # Create arrays for accumulating attention and counting overlaps
        norm_colors = np.zeros(seq_len)
        overlap_count = np.zeros(seq_len)
        
        # Accumulate attention values and count overlaps
        for i in range(n_patches):
            start_idx = i * stride
            end_idx = min(start_idx + patch_len, seq_len)  # Ensure we don't go beyond sequence length
            norm_colors[start_idx:end_idx] += attention_per_patch[i].cpu().numpy()
            overlap_count[start_idx:end_idx] += 1
            
        # Average the accumulated attention by the number of overlaps
        # Add small epsilon to avoid division by zero
        norm_colors = norm_colors / (overlap_count + 1e-8)
            
        # Normalize colors for visualization
        min_att = norm_colors.min()
        max_att = norm_colors.max()
        norm_colors = (norm_colors - min_att) / (max_att - min_att + 1e-8)
        
        # Create subplot for attention matrix
        ax1 = plt.subplot(n_rows, n_cols, row * n_cols + col + 1)

        # Plot signal with attention coloring
        cmap = plt.cm.viridis
        
        # Plot attention matrix
        cax1 = ax1.imshow(attention_matrix.cpu(), cmap=cmap, aspect='auto')
        ax1.set_title(f'Channel {channel_idx} - Attention Pattern')
        plt.colorbar(cax1, ax=ax1)
        
        # Create subplot for signal
        ax2 = plt.subplot(n_rows, n_cols, (row + 1) * n_cols + col + 1)
        
        # Create time axis for the full sequence length
        time_axis = np.arange(seq_len)
        val_axis = signal[:seq_len].cpu().numpy()  # Ensure signal matches sequence length

        scatter = ax2.scatter(time_axis, val_axis, c=norm_colors, cmap=cmap, s=10, alpha=0.8)
        ax2.plot(time_axis, val_axis, alpha=0.3, color='gray')
        
        # Add vertical lines for patch boundaries
        for p in range(n_patches + 1):
            boundary = min(p * stride, seq_len)
            if boundary < seq_len:  # Only draw lines within sequence length
                ax2.axvline(x=boundary, color='black', linestyle=':', alpha=0.5)
            
        # Set x-axis limits
        ax2.set_xlim(0, seq_len)
        
        # Add grid
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar for signal plot
        from matplotlib.cm import ScalarMappable
        import matplotlib.colors as mcolors
        sm = ScalarMappable(norm=mcolors.Normalize(min_att, max_att), cmap=cmap)
        sm.set_array([])
        plt.colorbar(sm, ax=ax2, label='Avg Attention Weight')
        
        ax2.set_title(f'Channel {channel_idx} - Signal')
    
    plt.tight_layout()
    if save:
         plt.savefig(save_dir)
    plt.close()

def visualize_attention_with_signal_head_channel_wise(fixed_train_sample, attention_per_channel, head_idx=0, sample_idx=0, patch_len=100, stride=100, save=False, save_dir=''):
    """
    Visualize attention patterns and signals for all channels for a specific attention head.
    
    Args:
        fixed_train_sample: tensor of shape [n_samples, n_patches, n_channels, patch_len]
        attention_per_channel: tensor of shape [n_samples, n_channels, n_heads, n_patches, n_patches]
        head_idx: which attention head to visualize
        sample_idx: which sample to visualize
        patch_len: length of each patch
        stride: stride between patches
    """
    n_channels = fixed_train_sample.shape[2]
    n_patches = fixed_train_sample.shape[1]
    n_heads = attention_per_channel.shape[2]
    
    # Validate head_idx
    if head_idx >= n_heads:
        raise ValueError(f"head_idx {head_idx} is out of range. Must be less than {n_heads}")
    
    # Calculate layout
    n_cols = min(12, n_channels)
    n_rows = math.ceil(n_channels / n_cols) * 2  # Multiply by 2 for attention + signal plots
    
    # Create figure
    fig = plt.figure(figsize=(5*n_cols, 4*n_rows))
    
    # For each channel
    for channel_idx in range(n_channels):
        # Calculate row and column position
        row = (channel_idx // n_cols) * 2  # Multiply by 2 to leave space for signal plot
        col = channel_idx % n_cols
        
        # Get signal for this channel
        signal = fixed_train_sample[sample_idx, :, channel_idx, :].flatten()
        
        # Get attention matrix for this channel and head
        attention_matrix = attention_per_channel[sample_idx, channel_idx, head_idx]  # Get specific head
        
        # Calculate attention received by each timestep
        attention_per_patch = attention_matrix.mean(dim=0)  # Average attention received by each patch
        
        # Calculate total sequence length
        seq_len = (n_patches - 1) * stride + patch_len
        
        # Create arrays for accumulating attention and counting overlaps
        norm_colors = np.zeros(seq_len)
        overlap_count = np.zeros(seq_len)
        
        # Accumulate attention values and count overlaps
        for i in range(n_patches):
            start_idx = i * stride
            end_idx = min(start_idx + patch_len, seq_len)  # Ensure we don't go beyond sequence length
            norm_colors[start_idx:end_idx] += attention_per_patch[i].cpu().numpy()
            overlap_count[start_idx:end_idx] += 1
            
        # Average the accumulated attention by the number of overlaps
        # Add small epsilon to avoid division by zero
        norm_colors = norm_colors / (overlap_count + 1e-8)
            
        # Normalize colors for visualization
        min_att = norm_colors.min()
        max_att = norm_colors.max()
        norm_colors = (norm_colors - min_att) / (max_att - min_att + 1e-8)
        
        # Create subplot for attention matrix
        ax1 = plt.subplot(n_rows, n_cols, row * n_cols + col + 1)
        
        # Plot signal with attention coloring
        cmap = plt.cm.viridis

        # Plot attention matrix
        cax1 = ax1.imshow(attention_matrix.cpu(), cmap=cmap, aspect='auto')
        ax1.set_title(f'Channel {channel_idx} - Head {head_idx} Attention')
        plt.colorbar(cax1, ax=ax1)
        
        # Create subplot for signal
        ax2 = plt.subplot(n_rows, n_cols, (row + 1) * n_cols + col + 1)
        
        # Create time axis for the full sequence length
        time_axis = np.arange(seq_len)
        val_axis = signal[:seq_len].cpu().numpy()  # Ensure signal matches sequence length
        
        scatter = ax2.scatter(time_axis, val_axis, c=norm_colors, cmap=cmap, s=10, alpha=0.8)
        ax2.plot(time_axis, val_axis, alpha=0.3, color='gray')
        
        # Add vertical lines for patch boundaries
        for p in range(n_patches + 1):
            boundary = min(p * stride, seq_len)
            if boundary < seq_len:  # Only draw lines within sequence length
                ax2.axvline(x=boundary, color='black', linestyle=':', alpha=0.5)
            
        # Set x-axis limits
        ax2.set_xlim(0, seq_len)
        
        # Add grid
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar for signal plot
        from matplotlib.cm import ScalarMappable
        import matplotlib.colors as mcolors
        sm = ScalarMappable(norm=mcolors.Normalize(min_att, max_att), cmap=cmap)
        sm.set_array([])
        plt.colorbar(sm, ax=ax2, label='Attention Weight')
        
        ax2.set_title(f'Channel {channel_idx} - Signal')
    
    plt.suptitle(f'Attention Patterns for Head {head_idx}', y=1.02)
    plt.tight_layout()
    if save:
         plt.savefig(save_dir)
    plt.close()

attention_weights = []
attention_hook = read_attention_hook_weights
num_layers = len(model.backbone.encoder.layers)


# make directory for attention plots
os.makedirs(f'experiment/{DATASET[:4]}', exist_ok=True)

for layer_idx in range(num_layers):

    # Print process 
    print(f'Processing layer {layer_idx} of {num_layers}')

    ATTENTION_CACHE = None

    # register the `read_attention_hook` on `model.backbone.encoder.layers[0].self_attn`
    model.backbone.encoder.layers[layer_idx].self_attn.register_forward_hook(attention_hook)

    # forward pass
    _ = plot_reconstruction(model, fixed_train_sample, MASK_RATIO, PATCH_LEN, STRIDE,
                            column_names, f'experiment/{DATASET[:4]}/reconstruction_for_att.png', revin, fixed_train_mask)
    
    # store the attention weights
    attention_weights.append(ATTENTION_CACHE)


samples = [0,1,2,3,4] # 1

for sample_idx in samples:
    for layer_idx in range(len(model.backbone.encoder.layers)):

        # Print process 
        print(f'Plotting layer {layer_idx} of {num_layers} for sample {sample_idx}')

        attention_at_layer = attention_weights[layer_idx]
        attention_per_channel = attention_at_layer.reshape(BATCH_SIZE, n_vars, N_HEADS, num_patches, num_patches)

        # save the attention plot for the average attention across all channels
        # make subdir for each layer
        os.makedirs(f'experiment/{DATASET[:4]}/layer_{layer_idx}/sample_{sample_idx}', exist_ok=True)
        visualize_attention_with_signal_head_average_channel_wise(fixed_train_sample, attention_per_channel, sample_idx=sample_idx, save=True, save_dir=f'experiment/{DATASET[:4]}/layer_{layer_idx}/sample_{sample_idx}/attention_average_channel_wise.png')

        # save the attention plot for the attention of each channel
        plot_n_heads = attention_weights[layer_idx].shape[1]
        #plot_n_heads = 16

        for head_idx in range(plot_n_heads):
            visualize_attention_with_signal_head_channel_wise(fixed_train_sample, attention_per_channel, head_idx=head_idx, sample_idx=sample_idx, save=True, save_dir=f'experiment/{DATASET[:4]}/layer_{layer_idx}/sample_{sample_idx}/attention_head_{head_idx}_channel_wise.png')






