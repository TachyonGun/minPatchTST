import torch
import torch.nn as nn
import torch.optim as optim
from data import create_dataloader
from models import create_patchtst_model, get_num_patches
from losses import MaskedMSELoss
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import wandb
import os

# Configuration
DATASET = 'seed_iv/session'
USE_WANDB = True # Set to True to enable wandb logging
CHECKPOINT_DIR = 'checkpoints'  # Directory to save model checkpoints


# Otherwise use one of these
#DATASET = 'your/path/to/dataset'  # Options: SEED, ETT-small, electricity, traffic, weather, or path to custom dataset
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
# Custom dataset configuration (if using GenericArrayDataset)
WEIGHT_DECAY = 0.0 # 1e-3  # Weight decay for regularization
DECAY_AFTER = None # None or number of epochs to decay after

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

USE_PATCH64 = False

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

elif USE_PATCH64:
    # Original hyperparameters for other datasets
    CONTEXT_POINTS = 512
    TARGET_POINTS = 96
    PATCH_LEN = 16
    STRIDE = 8
    BATCH_SIZE = 16
    MASK_RATIO = 0.4
    N_EPOCHS = 20
    D_MODEL = 128
    N_HEADS = 16
    D_FF = 256
    DROPOUT = 0.2
    HEAD_DROPOUT = 0.2
    USE_REVIN = True
    REVIN_AFFINE = True
    REVIN_EPS = 1e-5
    SUBTRACT_LAST = False

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

# Added learning rate scheduling parameters
MAX_LR = 5 * 1e-4 # 1e-4  # Peak learning rate for one-cycle
DIV_FACTOR = 1  # Initial learning rate will be MAX_LR/DIV_FACTOR
FINAL_DIV_FACTOR = 1  # Final learning rate will be MAX_LR/FINAL_DIV_FACTOR

# Optional visualization settings
PLOT_TRAIN_SAMPLE = True  # Set to True to plot a training sample at each epoch
PLOT_VAL_SAMPLE = True   # Set to True to plot a validation sample at each epoch
MAX_VARS_TO_PLOT = 6      # Maximum number of variables to plot
PLOT_FIGSIZE = (20, 10) # (15, 10)   # Figure size for plots
KEEP_CONSTANT_MASK = True  # If True, use the same mask pattern for visualization across epochs
COMPUTE_EPOCH_0_VALIDATION = True  # Set to True to compute and log validation loss at epoch 0

# Masking value for masked patches
MASKING_VALUE = 0.0  # Value to use for masked patches

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
    

def create_patch_mask(batch_size, num_patches, mask_ratio):
    """Create random mask for patches"""
    num_masked = int(num_patches * mask_ratio)
    mask = torch.zeros((batch_size, num_patches))
    for i in range(batch_size):
        # Randomly select patches to mask
        masked_indices = np.random.choice(num_patches, num_masked, replace=False)
        mask[i, masked_indices] = 1
    return mask.to(DEVICE)

def plot_sample_with_patches(data, output, mask, patch_len, stride, column_names=None, filename='sample_visualization.png'):
    """
    Plot a sample with its patches and model output
    Args:
        data: [batch_size, num_patches, n_vars, patch_len]
        output: [batch_size, num_patches, n_vars, patch_len]
        mask: [batch_size, num_patches]
        patch_len: patch length
        stride: stride between patches
        column_names: list of variable names
        filename: filename to save the plot
    """
    # Get the first batch
    sample_data = data[0].cpu().numpy()  # [num_patches, n_vars, patch_len]
    sample_output = output[0].cpu().numpy()  # [num_patches, n_vars, patch_len]
    sample_mask = mask[0].cpu().numpy()  # [num_patches]
    
    # Reconstruct the original sequence
    n_vars = sample_data.shape[1]
    num_patches = sample_data.shape[0]
    seq_len = (num_patches - 1) * stride + patch_len
    
    # Only plot up to MAX_VARS_TO_PLOT variables
    n_vars_to_plot = min(n_vars, MAX_VARS_TO_PLOT)
    
    # Setup figure
    plt.figure(figsize=PLOT_FIGSIZE)
    
    for var_idx in range(n_vars_to_plot):
        plt.subplot(n_vars_to_plot, 1, var_idx + 1)
        
        # Reconstruct original sequence for this variable
        original_seq = np.zeros(seq_len)
        count_seq = np.zeros(seq_len)
        model_output_seq = np.zeros(seq_len)
        
        # Fill in the sequences
        for p in range(num_patches):
            start_idx = p * stride
            end_idx = start_idx + patch_len
            patch_data = sample_data[p, var_idx]
            patch_output = sample_output[p, var_idx]
            
            # Original data
            original_seq[start_idx:end_idx] += patch_data
            count_seq[start_idx:end_idx] += 1
            
            # Mark the patches
            if sample_mask[p] == 1:  # If patch is masked
                # Model output (reconstruction)
                model_output_seq[start_idx:end_idx] += patch_output
                
                # Mark these points with different color
                plt.axvspan(start_idx, end_idx - 1, alpha=0.2, color='yellow', label='_nolegend_')
        
        # Average overlapping patches
        original_seq /= np.maximum(count_seq, 1)
        model_output_seq /= np.maximum(count_seq, 1)
        
        # Plot original sequence
        plt.plot(original_seq, label='Original', color='blue')
        
        # Create a flag to add reconstruction to legend only once
        added_to_legend = False
        
        # Plot masked regions that were reconstructed
        for p in range(num_patches):
            if sample_mask[p] == 1:  # If patch is masked
                start_idx = p * stride
                end_idx = start_idx + patch_len
                if not added_to_legend:
                    plt.plot(range(start_idx, end_idx), sample_output[p, var_idx], 
                            color='red', alpha=0.8, label='Reconstruction')
                    added_to_legend = True
                else:
                    plt.plot(range(start_idx, end_idx), sample_output[p, var_idx], 
                            color='red', alpha=0.8, label='_nolegend_')
        
        # Draw vertical lines at patch boundaries
        for p in range(num_patches):
            start_idx = p * stride
            plt.axvline(x=start_idx, color='black', linestyle=':', alpha=0.5, label='_nolegend_')
        
        # Last patch boundary
        plt.axvline(x=(num_patches-1)*stride + patch_len, color='black', linestyle=':', alpha=0.5, label='_nolegend_')
        
        # Variable name or index as title
        var_name = column_names[var_idx] if column_names is not None else f"Variable {var_idx+1}"
        plt.title(f"{var_name}")
        plt.grid(True, alpha=0.3)
        
        # Add legend with masked regions explanation
        if var_idx == 0:
            # Create custom legend elements
            from matplotlib.patches import Patch
            legend_elements = [
                plt.Line2D([0], [0], color='blue', label='Original'),
                plt.Line2D([0], [0], color='red', label='Reconstruction'),
                Patch(facecolor='yellow', alpha=0.2, label='Masked Regions')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def train_epoch(model, train_loader, optimizer, criterion, mask_ratio, scheduler=None, revin=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    last_batch = None
    
    # Wrap train_loader with tqdm
    pbar = tqdm(train_loader, desc='Training', leave=False)
    
    for batch_idx, (data, _) in enumerate(pbar):
        # Move data to device
        data = data.to(DEVICE)  # [batch_size, num_patches, n_vars, patch_len]
        batch_size = data.size(0)
        
        # Apply RevIN if used
        if revin is not None:
            data = revin(data, mode='norm')
        
        # Create mask
        mask = create_patch_mask(batch_size, data.size(1), mask_ratio)
        
        # Create masked input by replacing masked patches with MASKING_VALUE
        masked_data = data.clone()
        # Expand mask to match data dimensions [batch_size, num_patches, n_vars, patch_len]
        expanded_mask = mask.bool().unsqueeze(-1).unsqueeze(-1).expand_as(data)
        masked_data[expanded_mask] = MASKING_VALUE
        
        # Forward pass
        optimizer.zero_grad()
        output = model(masked_data)  # [batch_size, num_patch, n_vars, patch_len]
        
        # Calculate loss on masked patches
        loss = criterion(output, data, mask)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Step scheduler if using OneCycleLR
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        
        # Log training loss to wandb if enabled
        if USE_WANDB:
            wandb.log({"train/step_loss": loss.item()})
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # Save last batch for visualization
        if batch_idx == len(train_loader) - 1:
            last_batch = (data.detach(), output.detach(), mask.detach())
    
    return total_loss / len(train_loader), last_batch

def validate(model, val_loader, criterion, mask_ratio, revin=None):
    """Validate model"""
    model.eval()
    total_loss = 0
    last_batch = None
    
    # Wrap val_loader with tqdm
    pbar = tqdm(val_loader, desc='Validating', leave=False)
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(DEVICE)  # [batch_size, num_patches, n_vars, patch_len]
            batch_size = data.size(0)
            
            # Apply RevIN if used
            if revin is not None:
                data = revin(data, mode='norm')
            
            # Create mask
            mask = create_patch_mask(batch_size, data.size(1), mask_ratio)
            
            # Create masked input by replacing masked patches with MASKING_VALUE
            masked_data = data.clone()
            # Expand mask to match data dimensions [batch_size, num_patches, n_vars, patch_len]
            expanded_mask = mask.bool().unsqueeze(-1).unsqueeze(-1).expand_as(data)
            masked_data[expanded_mask] = MASKING_VALUE
            
            # Forward pass
            output = model(masked_data)
            
            # Calculate loss
            loss = criterion(output, data, mask)
            total_loss += loss.item()
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            # Save last batch for visualization
            if batch_idx == len(val_loader) - 1:
                last_batch = (data, output, mask)
            
    return total_loss / len(val_loader), last_batch

def get_fixed_sample(dataloader):
    """Get a fixed sample from dataloader for consistent visualization"""
    data_iter = iter(dataloader)
    data, _ = next(data_iter)
    return data.detach()

def get_fixed_mask(batch_size, num_patches, mask_ratio):
    """Create a fixed mask pattern for visualization"""
    num_masked = int(num_patches * mask_ratio)
    mask = torch.zeros((batch_size, num_patches))
    for i in range(batch_size):
        # Use a fixed seed for reproducibility
        rng_state = np.random.get_state()
        np.random.seed(42 + i)  # Different seed for each batch item but constant across epochs
        masked_indices = np.random.choice(num_patches, num_masked, replace=False)
        np.random.set_state(rng_state)
        mask[i, masked_indices] = 1
    return mask.to(DEVICE)

def plot_reconstruction(model, data, mask_ratio, patch_len, stride, column_names, filename, revin=None, fixed_mask=None):
    """Plot reconstruction for a fixed sample"""
    model.eval()
    with torch.no_grad():
        # Move data to device and create batch dimension if needed
        data = data.to(DEVICE)
        if len(data.shape) == 3:
            data = data.unsqueeze(0)
        
        # Apply RevIN if used
        if revin is not None:
            data = revin(data, mode='norm')
        
        # Create or use fixed mask
        if fixed_mask is not None:
            mask = fixed_mask
        else:
            mask = create_patch_mask(data.size(0), data.size(1), mask_ratio)
        
        # Create masked input by replacing masked patches with MASKING_VALUE
        masked_data = data.clone()
        # Expand mask to match data dimensions [batch_size, num_patches, n_vars, patch_len]
        expanded_mask = mask.bool().unsqueeze(-1).unsqueeze(-1).expand_as(data)
        masked_data[expanded_mask] = MASKING_VALUE
        
        # Get reconstruction
        output = model(masked_data)
        
        # Plot the results
        plot_sample_with_patches(data, output, mask, patch_len, stride, column_names, filename)
        
        # Log the figure to wandb if enabled
        # check if wandb was initialized
        if wandb.run is not None:
            wandb.log({filename: wandb.Image(filename)})

def main():
    # Create checkpoint directory if it doesn't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Initialize wandb if enabled
    if USE_WANDB:
        wandb.init(
            project="patchtst",
            name=f"{DATASET}_replication",
            config={
                "dataset": DATASET,
                "context_points": CONTEXT_POINTS,
                "target_points": TARGET_POINTS,
                "patch_len": PATCH_LEN,
                "stride": STRIDE,
                "batch_size": BATCH_SIZE,
                "mask_ratio": MASK_RATIO,
                "n_epochs": N_EPOCHS,
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "d_ff": D_FF,
                "dropout": DROPOUT,
                "head_dropout": HEAD_DROPOUT,
                "use_revin": USE_REVIN,
                "revin_affine": REVIN_AFFINE,
                "max_lr": MAX_LR,
                "div_factor": DIV_FACTOR,
                "final_div_factor": FINAL_DIV_FACTOR,
                "weight_decay": WEIGHT_DECAY,
            }
        )
    
    # Create dataloaders
    train_loader, val_loader, _, column_names = create_dataloader(
        DATASET,
        CONTEXT_POINTS,
        TARGET_POINTS,
        PATCH_LEN,
        STRIDE,
        batch_size=BATCH_SIZE
    )
    
    # Get fixed samples for visualization
    fixed_train_sample = get_fixed_sample(train_loader) if PLOT_TRAIN_SAMPLE else None
    fixed_val_sample = get_fixed_sample(val_loader) if PLOT_VAL_SAMPLE else None
    
    # Create fixed masks if needed
    fixed_train_mask = None
    fixed_val_mask = None
    if KEEP_CONSTANT_MASK:
        if fixed_train_sample is not None:
            fixed_train_mask = get_fixed_mask(fixed_train_sample.size(0), 
                                            get_num_patches(CONTEXT_POINTS, PATCH_LEN, STRIDE), 
                                            MASK_RATIO)
        if fixed_val_sample is not None:
            fixed_val_mask = get_fixed_mask(fixed_val_sample.size(0), 
                                          get_num_patches(CONTEXT_POINTS, PATCH_LEN, STRIDE), 
                                          MASK_RATIO)
    
    # Get input dimension (number of variables) and verify data format
    x_sample = fixed_train_sample if fixed_train_sample is not None else next(iter(train_loader))[0]
    n_vars = x_sample.size(2)  # [batch, num_patches, n_vars, patch_len]
    num_patches = x_sample.size(1)
    
    print(f"Data shape: {x_sample.shape}")
    print(f"Number of variables: {n_vars}")
    print(f"Number of patches: {num_patches}")
    print(f"Column names: {column_names}")
    print(f"Using {'constant' if KEEP_CONSTANT_MASK else 'random'} mask pattern for visualization")
    
    # Create model
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
    
    # Initialize RevIN if used
    revin = RevIN(n_vars, eps=REVIN_EPS, affine=REVIN_AFFINE, subtract_last=SUBTRACT_LAST) if USE_REVIN else None
    if revin is not None:
        revin = revin.to(DEVICE)
    
    # Plot initial reconstructions (epoch 0) before training
    print("\nPlotting initial reconstructions (epoch 0)...")
    if PLOT_TRAIN_SAMPLE and fixed_train_sample is not None:
        plot_reconstruction(model, fixed_train_sample, MASK_RATIO, PATCH_LEN, STRIDE, 
                          column_names, f'train_sample_epoch_0.png', revin, fixed_train_mask)
        
    if PLOT_VAL_SAMPLE and fixed_val_sample is not None:
        plot_reconstruction(model, fixed_val_sample, MASK_RATIO, PATCH_LEN, STRIDE,
                          column_names, f'val_sample_epoch_0.png', revin, fixed_val_mask)
    
    # Setup training
    criterion = MaskedMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=MAX_LR/DIV_FACTOR, weight_decay=WEIGHT_DECAY)
    
    # Setup OneCycleLR scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        epochs=N_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,  # Peak LR at 20% of training
        div_factor=DIV_FACTOR,
        final_div_factor=FINAL_DIV_FACTOR
    )
    
    # Compute validation loss at epoch 0 if enabled
    if COMPUTE_EPOCH_0_VALIDATION:
        print("\nComputing validation loss at epoch 0...")
        epoch_0_val_loss, _ = validate(model, val_loader, criterion, MASK_RATIO, revin)
        if USE_WANDB:
            wandb.log({
                "val/epoch_loss": epoch_0_val_loss,
                "epoch": 0
            })
        print(f'Epoch 0 Validation Loss: {epoch_0_val_loss:.6f}')
        print('-' * 50)
    
    # Training loop
    best_val_loss = float('inf')
    epoch_times = []  # Store epoch execution times
    
    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        
        train_loss, _ = train_epoch(model, train_loader, optimizer, criterion, MASK_RATIO, scheduler, revin)
        val_loss, _ = validate(model, val_loader, criterion, MASK_RATIO, revin)
        
        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        epoch_times.append(epoch_duration)
        
        # Calculate average epoch time and estimate remaining time
        avg_epoch_time = np.mean(epoch_times)
        epochs_remaining = N_EPOCHS - (epoch + 1)
        estimated_time_remaining = avg_epoch_time * epochs_remaining
        
        # Log metrics to wandb if enabled
        if USE_WANDB:
            wandb.log({
                "train/epoch_loss": train_loss,
                "val/epoch_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_duration": epoch_duration,
            })
        
        print(f'Epoch {epoch+1}/{N_EPOCHS}:')
        print(f'Train Loss: {train_loss:.6f}')
        print(f'Val Loss: {val_loss:.6f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'Epoch duration: {epoch_duration:.2f}s')
        
        # Format remaining time into hours, minutes, seconds
        hours = int(estimated_time_remaining // 3600)
        minutes = int((estimated_time_remaining % 3600) // 60)
        seconds = int(estimated_time_remaining % 60)
        print(f'Estimated time remaining: {hours:02d}:{minutes:02d}:{seconds:02d}')
        
        # Plot reconstructions using fixed samples
        if PLOT_TRAIN_SAMPLE and fixed_train_sample is not None:
            print("Plotting training sample...")
            plot_reconstruction(model, fixed_train_sample, MASK_RATIO, PATCH_LEN, STRIDE, 
                              column_names, f'train_sample_epoch_{epoch+1}.png', revin, fixed_train_mask)
            
        if PLOT_VAL_SAMPLE and fixed_val_sample is not None:
            print("Plotting validation sample...")
            plot_reconstruction(model, fixed_val_sample, MASK_RATIO, PATCH_LEN, STRIDE,
                              column_names, f'val_sample_epoch_{epoch+1}.png', revin, fixed_val_mask)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Create a safe filename by replacing potential problematic characters
            safe_dataset_name = DATASET.replace('/', '_').replace('\\', '_')
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'pretrained_{safe_dataset_name}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'revin_state_dict': revin.state_dict() if revin else None,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f'Model saved to {checkpoint_path}!')
        print('-' * 50)

        # Turn on weight decay after DECAY_AFTER epochs
        if DECAY_AFTER is not None and epoch >= DECAY_AFTER:
            scheduler.optimizer.param_groups[0]['weight_decay'] = WEIGHT_DECAY

    # Close wandb run if enabled
    if USE_WANDB:
        wandb.finish()

if __name__ == '__main__':
    main() 