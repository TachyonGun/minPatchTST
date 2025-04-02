import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    """Dataset for loading and preprocessing time series data"""
    def __init__(self, data, context_points, target_points, patch_len, stride, scale=True, column_names=None):
        super().__init__()
        self.data = data  # Shape: [seq_len, n_vars]
        self.context_points = context_points
        self.target_points = target_points
        self.patch_len = patch_len
        self.stride = stride
        self.scale = scale
        self.column_names = column_names  # Store column names
        
        if scale:
            # Standardize using numpy (equivalent to StandardScaler)
            self.mean = np.mean(self.data, axis=0, keepdims=True)
            self.std = np.std(self.data, axis=0, keepdims=True)
            self.std[self.std == 0] = 1  # Avoid division by zero
            self.data = (self.data - self.mean) / self.std
            
        # Calculate valid indices
        self.indices = self._get_valid_indices()
        
    def _get_valid_indices(self):
        """Get indices of all valid sequences"""
        total_len = self.context_points + self.target_points
        valid_indices = []
        for i in range(len(self.data) - total_len + 1):
            valid_indices.append(i)
        return valid_indices
    
    def _create_patches(self, sequence):
        """Convert sequence into patches
        Args:
            sequence: [context_points, n_vars]
        Returns:
            patches: [num_patches, n_vars, patch_len]
        """
        patches = []
        L = sequence.shape[0]  # sequence length
        for i in range(0, L - self.patch_len + 1, self.stride):
            patch = sequence[i:i + self.patch_len]  # [patch_len, n_vars]
            patches.append(patch.transpose(1, 0))  # [n_vars, patch_len]
        return torch.stack(patches)  # [num_patches, n_vars, patch_len]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.context_points + self.target_points
        
        # Get sequence
        sequence = self.data[start_idx:end_idx]
        
        # Split into input and target
        x = sequence[:self.context_points]
        y = sequence[self.context_points:]
        
        # Convert input into patches
        x = self._create_patches(torch.FloatTensor(x))  # [num_patches, n_vars, patch_len]
        y = torch.FloatTensor(y)
        
        return x, y

class SEEDTimeSeriesDataset(Dataset):
    """Dataset for loading and preprocessing SEED-IV EEG data"""
    def __init__(self, data_path, context_points, target_points, patch_len, stride):
        super().__init__()
        # Load data using memory mapping
        self.data = np.load(data_path, mmap_mode='r')  # Shape: [n_samples, n_channels, timesteps]
        self.context_points = context_points
        self.target_points = target_points
        self.patch_len = patch_len
        self.stride = stride
        
        # Verify we have enough timesteps for context + target
        total_points_needed = context_points + target_points
        if self.data.shape[2] < total_points_needed:
            raise ValueError(f"Segment length {self.data.shape[2]} is shorter than "
                           f"context ({context_points}) + target ({target_points}) points")
        
        # Calculate valid indices
        self.indices = self._get_valid_indices()
        
    def _get_valid_indices(self):
        """Get indices of all valid sequences within each segment"""
        total_len = self.context_points + self.target_points
        valid_indices = []
        
        # For each segment
        for seg_idx in range(len(self.data)):
            # Get number of valid starting points within this segment
            n_valid = self.data.shape[2] - total_len + 1
            # Add all valid (segment_idx, start_idx) pairs
            for start_idx in range(0, n_valid, self.stride):
                valid_indices.append((seg_idx, start_idx))
                
        return valid_indices
    
    def _create_patches(self, sequence):
        """Convert sequence into patches
        Args:
            sequence: [context_points, n_vars]
        Returns:
            patches: [num_patches, n_vars, patch_len]
        """
        patches = []
        L = sequence.shape[0]  # sequence length
        for i in range(0, L - self.patch_len + 1, self.stride):
            patch = sequence[i:i + self.patch_len]  # [patch_len, n_vars]
            patches.append(patch.transpose(1, 0))  # [n_vars, patch_len]
        return torch.stack(patches)  # [num_patches, n_vars, patch_len]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get segment index and start position
        seg_idx, start_idx = self.indices[idx]
        end_idx = start_idx + self.context_points + self.target_points
        
        # Get sequence from the segment
        # Memory mapping will only load this specific slice into RAM
        sequence = np.array(self.data[seg_idx, :, start_idx:end_idx])  # Copy to ensure contiguous memory
        
        # Split into input and target
        x = sequence[:, :self.context_points]  # [n_channels, context_points]
        y = sequence[:, self.context_points:]  # [n_channels, target_points]
        
        # Convert to torch tensors and transpose to match expected format
        x = torch.FloatTensor(x.T)  # [context_points, n_channels]
        y = torch.FloatTensor(y.T)  # [target_points, n_channels]
        
        # Create patches from input
        x = self._create_patches(x)  # [num_patches, n_channels, patch_len]
        
        return x, y

class GenericArrayDataset(Dataset):
    """Generic dataset for memory-mapped array data with optional train/val/test splits"""
    def __init__(self, data_path, context_points, target_points, patch_len, stride):
        super().__init__()
        # Load data using memory mapping
        self.data = np.load(data_path, mmap_mode='r')
        
        # Verify data shape: should be either (n_samples, channels, timesteps) or (channels, timesteps)
        if len(self.data.shape) == 2:
            self.data = self.data[np.newaxis, :]  # Add samples dimension
        elif len(self.data.shape) != 3:
            raise ValueError(f"Data should have shape (samples, channels, timesteps) or (channels, timesteps), "
                           f"got shape {self.data.shape}")
            
        self.context_points = context_points
        self.target_points = target_points
        self.patch_len = patch_len
        self.stride = stride
        
        # Verify we have enough timesteps for context + target
        total_points_needed = context_points + target_points
        if self.data.shape[2] < total_points_needed:
            raise ValueError(f"Segment length {self.data.shape[2]} is shorter than "
                           f"context ({context_points}) + target ({target_points}) points")
        
        # Calculate valid indices
        self.indices = self._get_valid_indices()
        
    def _get_valid_indices(self):
        """Get indices of all valid sequences within each segment"""
        total_len = self.context_points + self.target_points
        valid_indices = []
        
        # For each segment
        for seg_idx in range(len(self.data)):
            # Get number of valid starting points within this segment
            n_valid = self.data.shape[2] - total_len + 1
            # Add all valid (segment_idx, start_idx) pairs
            for start_idx in range(0, n_valid, self.stride):
                valid_indices.append((seg_idx, start_idx))
                
        return valid_indices
    
    def _create_patches(self, sequence):
        """Convert sequence into patches
        Args:
            sequence: [context_points, n_vars]
        Returns:
            patches: [num_patches, n_vars, patch_len]
        """
        patches = []
        L = sequence.shape[0]  # sequence length
        for i in range(0, L - self.patch_len + 1, self.stride):
            patch = sequence[i:i + self.patch_len]  # [patch_len, n_vars]
            patches.append(patch.transpose(1, 0))  # [n_vars, patch_len]
        return torch.stack(patches)  # [num_patches, n_vars, patch_len]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get segment index and start position
        seg_idx, start_idx = self.indices[idx]
        end_idx = start_idx + self.context_points + self.target_points
        
        # Get sequence from the segment
        sequence = np.array(self.data[seg_idx, :, start_idx:end_idx])  # Copy to ensure contiguous memory
        
        # Split into input and target
        x = sequence[:, :self.context_points]  # [n_channels, context_points]
        y = sequence[:, self.context_points:]  # [n_channels, target_points]
        
        # Convert to torch tensors and transpose to match expected format
        x = torch.FloatTensor(x.T)  # [context_points, n_channels]
        y = torch.FloatTensor(y.T)  # [target_points, n_channels]
        
        # Create patches from input
        x = self._create_patches(x)  # [num_patches, n_channels, patch_len]
        
        return x, y

def create_dataloader(dataset_name, context_points, target_points, patch_len, stride,
                     batch_size=32, scale=True, split_ratio=[0.7, 0.1, 0.2]):
    """Create train/val/test dataloaders"""
    # Special handling for SEED dataset
    if dataset_name == 'SEED':
        dataset_path = 'seed_iv/session'
        # Load train dataset
        train_dataset = GenericArrayDataset(
            os.path.join(dataset_path, 'train.npy'),
            context_points, target_points, patch_len, stride
        )
        
        # Load validation dataset
        val_dataset = GenericArrayDataset(
            os.path.join(dataset_path, 'validation.npy'),
            context_points, target_points, patch_len, stride
        )
        
        # Load test dataset
        test_dataset = GenericArrayDataset(
            os.path.join(dataset_path, 'test.npy'),
            context_points, target_points, patch_len, stride
        )
        
        # Get number of channels for column names
        n_channels = train_dataset.data.shape[1]
        column_names = [f'Channel_{i+1}' for i in range(n_channels)]
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, column_names
        
    # Check if directory contains train/val/test splits
    elif os.path.exists(os.path.join(dataset_name, 'train.npy')):
        # Load train dataset
        train_dataset = GenericArrayDataset(
            os.path.join(dataset_name, 'train.npy'),
            context_points, target_points, patch_len, stride
        )
        
        # Try to load validation dataset
        val_dataset = None
        if os.path.exists(os.path.join(dataset_name, 'validation.npy')):
            val_dataset = GenericArrayDataset(
                os.path.join(dataset_name, 'validation.npy'),
                context_points, target_points, patch_len, stride
            )
        elif os.path.exists(os.path.join(dataset_name, 'val.npy')):
            val_dataset = GenericArrayDataset(
                os.path.join(dataset_name, 'val.npy'),
                context_points, target_points, patch_len, stride
            )
            
        # Try to load test dataset
        test_dataset = None
        if os.path.exists(os.path.join(dataset_name, 'test.npy')):
            test_dataset = GenericArrayDataset(
                os.path.join(dataset_name, 'test.npy'),
                context_points, target_points, patch_len, stride
            )
        
        # Get number of channels for column names
        n_channels = train_dataset.data.shape[1]
        column_names = [f'Channel_{i+1}' for i in range(n_channels)]
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset else None
        
        return train_loader, val_loader, test_loader, column_names
    else:
        # Try to load single file 
        data_path = f'{dataset_name}.npy'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Could not find dataset at {data_path}. Please ensure the .npy file exists.")
            
        data = np.load(data_path)
        
        # Try to load column names from text file
        column_names_path = f'{dataset_name}_columns.txt'
        if os.path.exists(column_names_path):
            with open(column_names_path, 'r') as f:
                column_names = [line.strip() for line in f.readlines()]
            # Verify number of columns matches data
            if len(column_names) != data.shape[1]:
                print(f"Warning: Number of column names ({len(column_names)}) does not match data dimensions ({data.shape[1]})")
                # Fall back to default names
                column_names = [f'Feature_{i+1}' for i in range(data.shape[1])]
        else:
            column_names = [f'Feature_{i+1}' for i in range(data.shape[1])]
        
        # Split data
        n = len(data)
        train_end = int(n * split_ratio[0])
        val_end = int(n * (split_ratio[0] + split_ratio[1]))
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        # Create datasets
        train_dataset = TimeSeriesDataset(train_data, context_points, target_points, patch_len, stride, scale, column_names)
        val_dataset = TimeSeriesDataset(val_data, context_points, target_points, patch_len, stride, scale, column_names)
        test_dataset = TimeSeriesDataset(test_data, context_points, target_points, patch_len, stride, scale, column_names)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, column_names 