import os
import sys

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt
from patchtst.patchTST import PatchTST
from typing import List, Tuple
import pickle

class PredictionRecorder:
    def __init__(self, data: np.ndarray, time: np.ndarray):  # data: (3, 336), time: (336,)
        self.data = data
        self.time = time
        self.predictions: List[np.ndarray] = []  # Each prediction: (96, 3)
        self.losses: List[float] = []
        
    def add_prediction(self, pred: np.ndarray, loss: float):
        self.predictions.append(pred)
        self.losses.append(loss)
        
    def save(self, filename: str):
        save_dict = {
            'data': self.data,
            'time': self.time,
            'predictions': self.predictions,
            'losses': self.losses
        }
        with open(filename, 'wb') as f:
            pickle.dump(save_dict, f)
    
    @classmethod
    def load(cls, filename: str) -> 'PredictionRecorder':
        with open(filename, 'rb') as f:
            save_dict = pickle.load(f)
        recorder = cls(save_dict['data'], save_dict['time'])
        recorder.predictions = save_dict['predictions']
        recorder.losses = save_dict['losses']
        return recorder

def generate_fixed_data(n_samples=336, n_channels=3) -> Tuple[np.ndarray, np.ndarray]:
    """Generate fixed synthetic data with clear patterns"""
    t = np.linspace(0, 10, n_samples)  # (336,)
    data = []
    
    # Create three distinct patterns
    # Channel 1: Simple sine wave
    data.append(np.sin(2 * np.pi * 0.2 * t))
    
    # Channel 2: Sine wave with higher frequency
    data.append(np.sin(2 * np.pi * 0.5 * t))
    
    # Channel 3: Combined sine waves
    data.append(0.5 * np.sin(2 * np.pi * 0.3 * t) + 0.3 * np.sin(2 * np.pi * 0.6 * t))
    
    return np.array(data), t  # data: (3, 336), t: (336,)

def create_patches(data: np.ndarray, patch_len: int, stride: int) -> Tuple[np.ndarray, int]:
    """Convert time series into patches"""
    n_channels, n_samples = data.shape  # (3, 336)
    num_patches = (n_samples - patch_len) // stride + 1  # 41
    patches = np.zeros((num_patches, n_channels, patch_len))  # (41, 3, 16)
    
    for i in range(num_patches):
        start_idx = i * stride
        end_idx = start_idx + patch_len
        patches[i] = data[:, start_idx:end_idx]
    
    return patches, num_patches  # patches: (41, 3, 16), num_patches: 41

def print_shapes(data, time, patches, x, output, target, pred):
    """Print all relevant shapes in a clear format"""
    print("\nShape Information:")
    print("-" * 50)
    print(f"Original data shape (n_channels, n_samples): {data.shape}")
    print(f"Time array shape: {time.shape}")
    print(f"Patches shape (num_patches, n_channels, patch_len): {patches.shape}")
    print(f"Model input shape (batch, num_patches, n_channels, patch_len): {x.shape}")
    print(f"Model output shape (batch, target_len, n_channels): {output.shape}")
    print(f"Target shape (batch, target_len, n_channels): {target.shape}")
    print(f"Single prediction shape (target_len, n_channels): {pred.shape}")
    print("-" * 50)

def train_to_overfit_save_preds():
    # Model parameters
    patch_len = 16
    stride = 8
    d_model = 128
    n_heads = 16
    n_layers = 3
    
    # Generate fixed data
    data, time = generate_fixed_data()  # data: (3, 336), time: (336,)
    patches, num_patches = create_patches(data, patch_len, stride)  # patches: (41, 3, 16)
    
    # Initialize recorder
    recorder = PredictionRecorder(data, time)
    
    # Create model
    model = PatchTST(
        c_in=3,               # number of input channels
        target_dim=96,        # prediction horizon
        patch_len=patch_len,  # 16
        stride=stride,        # 8
        num_patch=num_patches,# 41
        n_layers=n_layers,    # 3
        d_model=d_model,      # 128
        n_heads=n_heads,      # 16
        head_type="prediction"
    )
    
    # Prepare input data
    x = torch.FloatTensor(patches)  # (41, 3, 16)
    x = x.unsqueeze(0)  # Add batch dimension -> (1, 41, 3, 16)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Create target: next 96 points after the sequence
    t_future = np.linspace(10, 13, 96)  # (96,)
    target = np.zeros((1, 96, 3))  # (1, 96, 3)
    target[0, :, 0] = np.sin(2 * np.pi * 0.2 * t_future)
    target[0, :, 1] = np.sin(2 * np.pi * 0.5 * t_future)
    target[0, :, 2] = 0.5 * np.sin(2 * np.pi * 0.3 * t_future) + 0.3 * np.sin(2 * np.pi * 0.6 * t_future)
    target = torch.FloatTensor(target)  # (1, 96, 3)
    
    # Print shapes before training
    with torch.no_grad():
        output = model(x)  # (1, 96, 3)
        pred = output.numpy()[0]  # (96, 3)
        print_shapes(data, time, patches, x, output, target, pred)
    
    # Training loop
    n_epochs = 100
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(x)  # (1, 96, 3)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Save prediction
        with torch.no_grad():
            pred = output.numpy()[0]  # (96, 3)
            recorder.add_prediction(pred, loss.item())
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    # Save the predictions
    recorder.save('prediction_history.pkl')
    return recorder

def create_prediction_plots(recorder: PredictionRecorder):
    """Create and save prediction plots for each epoch in test_results directory"""
    # Create test_results directory if it doesn't exist
    os.makedirs('test_results', exist_ok=True)
    
    # Calculate fixed axis limits for all plots
    pred_time = np.linspace(recorder.time[-1], 
                           recorder.time[-1] + recorder.time[-1]/3, 
                           recorder.predictions[0].shape[0])  # (96,)
    
    # X-axis limits
    x_min = min(recorder.time.min(), pred_time.min())
    x_max = max(recorder.time.max(), pred_time.max())
    x_range = x_max - x_min
    x_margin = x_range * 0.02  # 2% margin
    x_limits = [x_min - x_margin, x_max + x_margin]
    
    # Y-axis limits for each channel
    y_limits = []
    for i in range(recorder.data.shape[0]):
        # Get min/max from both original data and all predictions for this channel
        data_min = recorder.data[i].min()
        data_max = recorder.data[i].max()
        pred_min = min(pred[:, i].min() for pred in recorder.predictions)
        pred_max = max(pred[:, i].max() for pred in recorder.predictions)
        
        y_min = min(data_min, pred_min)
        y_max = max(data_max, pred_max)
        y_range = y_max - y_min
        y_margin = y_range * 0.05  # 5% margin
        y_limits.append([y_min - y_margin, y_max + y_margin])
    
    for epoch in range(len(recorder.predictions)):
        plt.figure(figsize=(15, 10))
        
        pred = recorder.predictions[epoch]  # (96, 3)
        
        for i in range(recorder.data.shape[0]):
            plt.subplot(recorder.data.shape[0], 1, i+1)
            
            # Plot original data
            plt.plot(recorder.time, recorder.data[i], 
                    label='Original', color='blue', alpha=0.5)
            
            # Plot prediction
            plt.plot(pred_time, pred[:, i], 
                    label=f'Prediction (Loss: {recorder.losses[epoch]:.6f})', 
                    color='red', linestyle='--')
            
            plt.title(f'Channel {i+1} - Epoch {epoch}')
            plt.grid(True)
            plt.legend()
            
            # Set fixed axis limits
            plt.xlim(x_limits)
            plt.ylim(y_limits[i])
        
        plt.tight_layout()
        plt.savefig(f'test_results/prediction_epoch_{epoch:03d}.png')
        plt.close()
        
        if epoch % 10 == 0:
            print(f"Saved plot for epoch {epoch}")

def load_and_visualize():
    """Load saved predictions and create visualizations"""
    recorder = PredictionRecorder.load('prediction_history.pkl')
    create_prediction_plots(recorder)

if __name__ == "__main__":
    # Train and save predictions
    recorder = train_to_overfit_save_preds()
    
    # Create and save visualizations
    create_prediction_plots(recorder)