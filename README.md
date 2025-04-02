# minPatchTST

This project is an independent validation of PatchTST (A Time Series is Worth 64 Words: Long-term Forecasting with Transformers). It reuses the model code from the [original GitHub implementation](https://github.com/yuqinie98/PatchTST) to support additional SSL investigations.

## Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/TachyonGun/minPatchTST.git
cd minPatchTST
pip install wandb twdm matplotlib numpy torch
```

## Quickstart

![Overfitting Demonstration](assets/output.gif)

To quickly validate the model's capability, run the overfitting test:

```bash
python tests/test_overfit.py
```

This script generates synthetic time series data with clear patterns (sine waves) and demonstrates the model's ability to learn and predict these patterns. The test creates visualizations in `test_results/` showing how the model learns to predict future values.


# Pretraining on Custom, Paper and SEED IV Datasets

![Weather Dataset](assets/weather.png)
*Weather dataset reconstructions*

To pretrain on one of the datasets used in the original paper:

1. Download the dataset files from <https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy>
2. Place all CSV files in a `datasets/` folder in the project root
3. Run the pretraining script:
   ```bash
   python pretrain.py
   ```

You can modify the `DATASET` variable in `pretrain.py` to choose from any of these datasets:

- **⭐Your dataset⭐** (See below)
- ETTh1 (If you did the above)
- ETTh2 (")
- ETTm1 (")
- ETTm2 (")
- electricity (")
- traffic (")
- weather (")
- SEED (See below)

## Using Custom Datasets

The project now supports any time series dataset stored as numpy arrays with a simple folder structure. Just organize your data as follows:

```
 my_dataset/
    ├── train.npy      # Required: shape (samples, channels, timesteps) or (channels, timesteps)
    ├── validation.npy # Optional!!!: same shape structure as train
    └── test.npy      # Optional!!!: same shape structure as train
```

The shape can be, for example:
 - "10 samples of 1000 time steps for 16 channels" (10, 16, 1000), 
 - a "10000 time steps for 16 channels" if you want to use it as one contiguous array.

The data loader will automatically:
- Use memory mapping for efficient loading
- Handle both 2D (channels, timesteps) and 3D (segments, channels, timesteps) arrays
- Load validation and test sets if available
- Create appropriate data loaders for the PatchTST model

Example usage:

```python
from data import create_dataloader

# Load your dataset
train_loader, val_loader, test_loader, columns = create_dataloader(
    'my_dataset',      # Will look in /my_dataset/ or whatever path you choose
    context_points=1000,
    target_points=200,
    patch_len=100,
    stride=50
)

# If validation.npy and test.npy don't exist, 
# val_loader and test_loader will be None
```

You can also use alternative folder structures by modifying the dataset name:

```python
# Load from a different directory
loaders = create_dataloader('seed_iv/session/', ...)

# The SEED dataset structure the code below creates
seed_iv/
└── session/
    ├── train.npy
    ├── validation.npy
    └── test.npy
```

*This makes it easy to use the PatchTST model with any time series data without modifying the codebase*. Just save your preprocessed data in the expected format and start training!

Note: The data should be preprocessed and normalized before saving as .npy files. The model expects numerical time series data where each channel represents a different feature or measurement over time and *that's it*, it's completelty generic.

## Setting up the SEED Dataset

![SEED IV Training](assets/train_sample_epoch_2.png)

*Seed IV subsample reconstructions after just 2 epochs*

1. Download [SEED-IV subsample off Kaggle](https://www.kaggle.com/datasets/phhasian0710/seed-iv)

2. Place your SEED-IV dataset folder in the root directory with the following structure:
```
.
├── seed_iv/
│   ├── eeg_raw_data/
│   │   ├── 1/
│   │   │   ├── 1_20160518.mat
│   │   │   ├── 1_20160518_blink.mat
│   │   │   └── ...
│   │   ├── 2/
│   │   └── ...
│   └── emotion_labels/
│       ├── 1_20160518.mat
│       └── ...
```

3. Run the preprocessing script:
```bash
python make_seed.py
```

This will create the processed dataset in the following structure:
```
.
├── seed_iv/
│   ├── session/
│   │   ├── train.npy  # Training data
│   │   ├── val.npy    # Validation data
│   │   └── test.npy   # Test data
│   └── ... (original data folders)
```

The processed .npy files will contain the EEG data in the format expected by the model.

## TODO

- Validate with downstream evaluations
- Support for TUH data loading


