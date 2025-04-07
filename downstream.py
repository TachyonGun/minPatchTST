import torch
import torch.nn as nn
import torch.optim as optim
from data import create_classification_dataloader
from models import create_patchtst_model, get_num_patches
import numpy as np
import os
from tqdm import tqdm

# Enable TF32 and automatic mixed precision
torch.set_float32_matmul_precision('high')

# Set dynamo config to handle errors gracefully
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # Fallback to eager mode if compilation fails

# Configuration
CHECKPOINT_PATH = 'checkpoints/pretrained_seed_iv_session.pth'
DATASET = 'seed_iv/session'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABELS = ['neutral', 'sad', 'fear', 'happy']  # Labels 0, 1, 2, 3 respectively
LINEAR_PROBE = False  # If False, finetune the entire model
USE_WANDB = True    # Set to True to enable wandb logging

# Import wandb only if enabled
if USE_WANDB:
    import wandb

# Use the same config as pretraining for consistency
GENERIC_CONFIG = {
    'context_points': 2000,    # Number of input timesteps
    'target_points': len(LABELS),  # Number of classes
    'patch_len': 200,          # Length of each patch
    'stride': 100,              # Stride between patches
    'batch_size': int(64),         # Batch size for training
    'd_model': 768 // 2,           # Model dimension
    'n_heads': 6,             # Number of attention heads
    'd_ff': 768 * 2,             # Feed-forward dimension
    'dropout': 0.05,          # Dropout rate
    'head_dropout': 0.01,     # Head dropout rate
    'n_epochs': 10,          # Number of training epochs
    'learning_rate': 2e-4,    # Learning rate
}

def compute_metrics(preds, labels):
    """Compute accuracy metrics"""
    preds = preds.argmax(dim=1)
    total_acc = (preds == labels).float().mean().item()
    
    # Per-class accuracy
    acc_per_class = []
    for i in range(len(LABELS)):
        mask = labels == i
        if mask.sum() > 0:
            class_acc = (preds[mask] == labels[mask]).float().mean().item()
            acc_per_class.append(class_acc)
        else:
            acc_per_class.append(0.0)
            
    return total_acc, acc_per_class

def evaluate(model, val_loader):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, labels in val_loader:  # Remove unused target from unpacking
            x, labels = x.to(DEVICE), labels.to(DEVICE).long()  # Ensure labels are long tensors
            output = model(x)  # [batch_size x n_classes]
            loss = criterion(output, labels)
            total_loss += loss.item()
            
            all_preds.append(output)
            all_labels.append(labels)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    acc, acc_per_class = compute_metrics(all_preds, all_labels)
    
    return total_loss / len(val_loader), acc, acc_per_class

def main():
    # Initialize wandb if enabled
    if USE_WANDB:
        wandb.init(
            project="patchtst-downstream",
            name=f"{'linear_probe' if LINEAR_PROBE else 'finetune'}_{DATASET}",
            config=GENERIC_CONFIG
        )
    
    # Create dataloaders
    train_loader, val_loader, test_loader, column_names = create_classification_dataloader(
        DATASET,
        GENERIC_CONFIG['context_points'],
        GENERIC_CONFIG['target_points'],
        GENERIC_CONFIG['patch_len'],
        GENERIC_CONFIG['stride'],
        batch_size=GENERIC_CONFIG['batch_size']
    )
    
    # Get model dimensions from data
    x_sample, _ = next(iter(train_loader))  # Remove unused target from unpacking
    n_vars = x_sample.size(2)  # [batch, num_patches, n_vars, patch_len]
    num_patches = x_sample.size(1)
    
    print(f"Data shape: {x_sample.shape}")
    print(f"Number of variables: {n_vars}")
    print(f"Number of patches: {num_patches}")
    
    # Create model
    model = create_patchtst_model(
        c_in=n_vars,
        target_dim=len(LABELS),  # Number of classes
        patch_len=GENERIC_CONFIG['patch_len'],
        stride=GENERIC_CONFIG['stride'],
        num_patch=num_patches,
        n_layers=6,
        d_model=GENERIC_CONFIG['d_model'],
        n_heads=GENERIC_CONFIG['n_heads'],
        d_ff=GENERIC_CONFIG['d_ff'],
        dropout=GENERIC_CONFIG['dropout'],
        head_dropout=GENERIC_CONFIG['head_dropout'],
        head_type="classification"  # Explicitly set head type for classification
    ).to(DEVICE)
    
    # Load pretrained weights
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # Only load backbone weights, skip the head
    pretrained_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    
    # Filter out head parameters
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.startswith('backbone')}
    
    # Update only the backbone parameters
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    # Freeze backbone if doing linear probe
    if LINEAR_PROBE:
        for param in model.backbone.parameters():
            param.requires_grad = False
            
    # # Enable graph compilation for faster training after loading weights
    # try:
    #     # Try using a more stable backend
    #     model = torch.compile(model, mode="reduce-overhead", backend="eager")
    # except Exception as e:
    #     print(f"Warning: Model compilation failed, falling back to eager mode. Error: {e}")
    #     pass
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters() if not LINEAR_PROBE else model.head.parameters(),
        lr=GENERIC_CONFIG['learning_rate']
    )
    
    # Training loop
    best_val_acc = 0
    for epoch in range(GENERIC_CONFIG['n_epochs']):
        model.train()
        total_loss = 0
        all_train_preds = []
        all_train_labels = []
        
        # Training
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{GENERIC_CONFIG["n_epochs"]}')
        for x, labels in pbar:  # Remove unused target from unpacking
            x, labels = x.to(DEVICE), labels.to(DEVICE).long()  # Ensure labels are long tensors
            
            optimizer.zero_grad()
            output = model(x)  # [batch_size x n_classes]
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            all_train_preds.append(output.detach())
            all_train_labels.append(labels)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute training metrics
        all_train_preds = torch.cat(all_train_preds, dim=0)
        all_train_labels = torch.cat(all_train_labels, dim=0)
        train_acc, train_acc_per_class = compute_metrics(all_train_preds, all_train_labels)
        
        # Validation
        val_loss, val_acc, val_acc_per_class = evaluate(model, val_loader)
        
        # Prepare metrics dict
        metrics = {
            'train/loss': total_loss / len(train_loader),
            'train/acc': train_acc,
            'val/loss': val_loss,
            'val/acc': val_acc,
        }
        
        # Add per-class accuracies
        for i, label in enumerate(LABELS):
            metrics[f'train/acc_{label}'] = train_acc_per_class[i]
            metrics[f'val/acc_{label}'] = val_acc_per_class[i]
        
        # Log metrics to wandb if enabled
        if USE_WANDB:
            wandb.log(metrics)
        
        # Print metrics
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {metrics["train/loss"]:.4f}, Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
        print('Train Acc per class:', {LABELS[i]: f'{acc:.4f}' for i, acc in enumerate(train_acc_per_class)})
        print('Val Acc per class:', {LABELS[i]: f'{acc:.4f}' for i, acc in enumerate(val_acc_per_class)})
        print('-' * 80)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f'checkpoints/downstream_{DATASET.replace("/", "_")}.pth')
    
    # Final test set evaluation
    test_loss, test_acc, test_acc_per_class = evaluate(model, test_loader)
    print('\nTest Results:')
    print(f'Loss: {test_loss:.4f}, Acc: {test_acc:.4f}')
    print('Acc per class:', {LABELS[i]: f'{acc:.4f}' for i, acc in enumerate(test_acc_per_class)})
    
    # Log final test metrics to wandb if enabled
    if USE_WANDB:
        wandb.log({
            'test/loss': test_loss,
            'test/acc': test_acc,
            **{f'test/acc_{label}': acc for label, acc in zip(LABELS, test_acc_per_class)}
        })
        wandb.finish()

if __name__ == '__main__':
    main() 