import torch
import matplotlib.pyplot as plt
import numpy as np
from models import create_patchtst_model
import os

def plot_matrix(matrix, title, filename, vmin=None, vmax=None):
    """Plot a matrix as a heatmap"""
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix.detach().cpu().numpy(), cmap='RdBu', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    # Add axis labels
    plt.xlabel('Output dimension')
    plt.ylabel('Input dimension')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def plot_concatenated_matrices(matrices, title, filename, n_heads_per_layer, n_layers, vmin=None, vmax=None):
    """Plot matrices concatenated vertically by heads and horizontally by layers"""
    # Get dimensions
    d_k = matrices[0].shape[0]
    d_model = matrices[0].shape[1]
    
    # Create a large figure
    total_height = d_k * n_heads_per_layer
    total_width = d_model * n_layers
    
    # Scale figure size to maintain aspect ratio but not be too large
    scale_factor = min(20 / total_width, 2)  # Cap width at 20 inches
    plt.figure(figsize=(total_width * scale_factor, total_height * scale_factor))
    
    # Create the combined matrix
    combined = torch.zeros(total_height, total_width)
    
    # Fill in the matrices
    for layer in range(n_layers):
        for head in range(n_heads_per_layer):
            matrix_idx = layer * n_heads_per_layer + head
            start_row = head * d_k
            end_row = start_row + d_k
            start_col = layer * d_model
            end_col = start_col + d_model
            combined[start_row:end_row, start_col:end_col] = matrices[matrix_idx]
    
    # Plot
    plt.imshow(combined.detach().cpu().numpy(), cmap='RdBu', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar()
    
    # Add grid lines to separate layers (vertical lines)
    for i in range(1, n_layers):
        plt.axvline(x=i*d_model-0.5, color='black', linestyle='--', alpha=0.3)
    
    # Add grid lines to separate heads (horizontal lines)
    for i in range(1, n_heads_per_layer):
        plt.axhline(y=i*d_k-0.5, color='black', linestyle='--', alpha=0.3)
    
    # Add labels
    plt.xlabel('Layer dimension (d_model) × Number of layers')
    plt.ylabel('Head dimension (d_k) × Number of heads')
    plt.title(title)
    
    # Save with high quality
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def plot_mlp_weights(mlp_weights, title, filename, n_layers, vmin=None, vmax=None):
    """Plot MLP weights concatenated horizontally by layers"""
    # Get dimensions
    d_in = mlp_weights[0].shape[0]
    d_out = mlp_weights[0].shape[1]
    
    # Create a large figure
    total_width = d_out * n_layers
    
    # Scale figure size to maintain aspect ratio but not be too large
    scale_factor = min(20 / total_width, 2)  # Cap width at 20 inches
    plt.figure(figsize=(total_width * scale_factor, d_in * scale_factor))
    
    # Create the combined matrix
    combined = torch.zeros(d_in, total_width)
    
    # Fill in the matrices
    for layer in range(n_layers):
        start_col = layer * d_out
        end_col = start_col + d_out
        combined[:, start_col:end_col] = mlp_weights[layer]
    
    # Plot
    plt.imshow(combined.detach().cpu().numpy(), cmap='RdBu', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar()
    
    # Add grid lines to separate layers
    for i in range(1, n_layers):
        plt.axvline(x=i*d_out-0.5, color='black', linestyle='--', alpha=0.3)
    
    # Add labels
    plt.xlabel('Output dimension × Number of layers')
    plt.ylabel('Input dimension')
    plt.title(title)
    
    # Save with high quality
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Create output directory
    os.makedirs('weight_visualizations', exist_ok=True)
    
    # Load the pretrained model
    checkpoint = torch.load('checkpoints/pretrained_seed_iv_session.pth', map_location='cpu')
    
    # Get model parameters from checkpoint
    state_dict = checkpoint['model_state_dict']
    
    # Create model with same architecture as in checkpoint
    model = create_patchtst_model(
        c_in=62,  # Number of input variables
        target_dim=200,  # For pretraining, this should match patch_len
        patch_len=200,  # From checkpoint
        stride=100,  # From default in pretrain.py
        num_patch=19,  # Will be determined by the data
        n_layers=6,  # Default from pretrain.py
        d_model=768 // 2,  # Default from pretrain.py
        n_heads=6,  # Default from pretrain.py
        d_ff= 768 * 2,  # From checkpoint
        dropout=0.2,  # Default from pretrain.py
        head_dropout=0.2  # Default from pretrain.py
    )
    
    # Load weights
    model.load_state_dict(state_dict)
    
    # Extract embedding matrix (W_P)
    embedding_matrix = model.backbone.W_P.weight
    plot_matrix(embedding_matrix, 'Embedding Matrix (W_P)', 'weight_visualizations/embedding_matrix.png')
    
    # Extract unembedding matrix (head linear layer)
    unembedding_matrix = model.head.linear.weight
    plot_matrix(unembedding_matrix, 'Unembedding Matrix (Head)', 'weight_visualizations/unembedding_matrix.png')
    
    # Extract attention matrices for each layer and head
    n_layers = len(model.backbone.encoder.layers)
    n_heads = model.backbone.encoder.layers[0].self_attn.n_heads
    
    # Lists to store matrices
    query_matrices = []
    key_matrices = []
    value_matrices = []
    
    # Lists to store MLP weights
    mlp_weights_1 = []  # First layer of each MLP
    mlp_weights_2 = []  # Second layer of each MLP
    
    # Extract matrices for each layer and head
    for layer in range(n_layers):
        attn = model.backbone.encoder.layers[layer].self_attn
        
        # Get the full weight matrices
        W_Q = attn.W_Q.weight  # [d_k * n_heads, d_model]
        W_K = attn.W_K.weight  # [d_k * n_heads, d_model]
        W_V = attn.W_V.weight  # [d_v * n_heads, d_model]
        
        # Get MLP weights
        ff = model.backbone.encoder.layers[layer].ff
        mlp_weights_1.append(ff[0].weight)  # First linear layer [d_ff, d_model]
        mlp_weights_2.append(ff[3].weight)  # Second linear layer [d_model, d_ff]
        
        # Split into per-head matrices
        d_k = attn.d_k
        d_v = attn.d_v
        
        # Reshape the matrices to separate heads
        W_Q = W_Q.view(n_heads, d_k, -1)  # [n_heads, d_k, d_model]
        W_K = W_K.view(n_heads, d_k, -1)  # [n_heads, d_k, d_model]
        W_V = W_V.view(n_heads, d_v, -1)  # [n_heads, d_v, d_model]
        
        for head in range(n_heads):
            query_matrices.append(W_Q[head])
            key_matrices.append(W_K[head])
            value_matrices.append(W_V[head])
    
    # Plot concatenated matrices
    # Find global min/max for consistent color scaling across Q,K,V
    all_matrices = torch.cat([
        torch.cat(query_matrices), 
        torch.cat(key_matrices), 
        torch.cat(value_matrices)
    ], dim=0)
    vmin, vmax = float(all_matrices.min()), float(all_matrices.max())
    print(key_matrices)
    # Ensure symmetric color range around 0
    abs_max = max(abs(vmin), abs(vmax))
    vmin_q, vmax_q = float(torch.cat(query_matrices).min()), float(torch.cat(query_matrices).max())
    vmin_k, vmax_k = float(torch.cat(key_matrices).min()), float(torch.cat(key_matrices).max())
    vmin_v, vmax_v = float(torch.cat(value_matrices).min()), float(torch.cat(value_matrices).max())
    
    plot_concatenated_matrices(query_matrices, 'Query Matrices (All Layers and Heads)', 
                             'weight_visualizations/query_matrices.png', n_heads, n_layers, vmin_q, vmax_q)
    plot_concatenated_matrices(key_matrices, 'Key Matrices (All Layers and Heads)', 
                             'weight_visualizations/key_matrices.png', n_heads, n_layers, vmin_k, vmax_k)
    plot_concatenated_matrices(value_matrices, 'Value Matrices (All Layers and Heads)', 
                             'weight_visualizations/value_matrices.png', n_heads, n_layers, vmin_v, vmax_v)
    
    # Plot MLP weights
    # Plot first layer and second layer weights separately since they have different dimensions
    # First layer: d_ff x d_model
    vmin1, vmax1 = float(torch.cat(mlp_weights_1).min()), float(torch.cat(mlp_weights_1).max())
    abs_max1 = max(abs(vmin1), abs(vmax1))
    vmin1, vmax1 = -abs_max1, abs_max1
    
    # Second layer: d_model x d_ff
    vmin2, vmax2 = float(torch.cat(mlp_weights_2).min()), float(torch.cat(mlp_weights_2).max())
    abs_max2 = max(abs(vmin2), abs(vmax2))
    vmin2, vmax2 = -abs_max2, abs_max2
    
    plot_mlp_weights(mlp_weights_1, 'MLP First Layer Weights (d_ff × d_model)', 
                    'weight_visualizations/mlp_layer1_weights.png', n_layers, vmin1, vmax1)
    plot_mlp_weights(mlp_weights_2, 'MLP Second Layer Weights (d_model × d_ff)', 
                    'weight_visualizations/mlp_layer2_weights.png', n_layers, vmin2, vmax2)

if __name__ == '__main__':
    main() 