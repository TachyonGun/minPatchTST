import os
import numpy as np
import scipy.io as sio
from tqdm import tqdm

def get_session_labels(session_id):
    """Return emotion labels for a given session"""
    labels = {
        1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
        2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
        3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    }
    return labels[session_id]

def process_session_data(session_id, normalize_95th=True):
    """Process one session's data and return concatenated array with labels"""
    session_dir = f"seed_iv/eeg_raw_data/{session_id}"
    files = sorted([f for f in os.listdir(session_dir) if f.endswith('.mat')])
    
    all_data = []
    all_labels = []
    recording_ids = []  # Track which recording each segment came from
    session_labels = get_session_labels(session_id)
    
    print(f"\nProcessing session {session_id}")
    recording_counter = 0
    for file in tqdm(files):
        try:
            mat_data = sio.loadmat(os.path.join(session_dir, file))
            
            # Find the EEG prefix
            eeg_prefix = None
            for key in mat_data.keys():
                if key.endswith('eeg1'):
                    eeg_prefix = key[:-1]
                    break
            
            if not eeg_prefix:
                print(f"\nWarning: Could not find EEG prefix in {file}")
                continue
                
            # Process each trial
            for trial_idx, label in enumerate(session_labels):
                key = f"{eeg_prefix}{trial_idx + 1}"
                if key not in mat_data:
                    print(f"\nWarning: {key} not found in {file}")
                    continue
                
                data = mat_data[key]  # Shape: [n_channels, timesteps]
                
                # Split into 10-second segments (10000 timesteps at 1000Hz)
                segment_length = 2000
                n_complete_segments = data.shape[1] // segment_length
                
                # Store all segments from this recording together
                recording_segments = []
                for i in range(n_complete_segments):
                    start_idx = i * segment_length
                    end_idx = start_idx + segment_length
                    segment = data[:, start_idx:end_idx]
                    recording_segments.append(segment)
                
                if recording_segments:
                    all_data.append(np.stack(recording_segments))  # Shape: [n_segments, n_channels, timesteps]
                    all_labels.extend([label] * len(recording_segments))
                    recording_ids.extend([recording_counter] * len(recording_segments))
                    recording_counter += 1
                    
        except Exception as e:
            print(f"\nError processing {file}: {str(e)}")
            continue
    
    # Convert to numpy arrays
    data_array = np.concatenate(all_data, axis=0)  # Shape: [total_segments, n_channels, timesteps]
    labels_array = np.array(all_labels)  # Shape: [total_segments]
    recording_ids = np.array(recording_ids)  # Shape: [total_segments]
    
    if normalize_95th:
        # Normalize each channel by its 95th percentile
        percentile_95 = np.percentile(np.abs(data_array), 95, axis=(0, 2), keepdims=True)
        percentile_95[percentile_95 == 0] = 1.0
        data_array = data_array / percentile_95
    
    return data_array, labels_array, recording_ids

def save_train_val_test_split(save_dir='seed_iv/session1', val_ratio=0.5):
    """
    Process all sessions and create train/val/test split:
    - Sessions 1 & 2: training
    - Session 3: split between validation and test
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Process training data (sessions 1 & 2)
    print("\nProcessing training data...")
    train_data_1, train_labels_1, train_rec_1 = process_session_data(session_id=1)
    train_data_2, train_labels_2, train_rec_2 = process_session_data(session_id=2)
    
    # Combine training data
    train_data = np.concatenate([train_data_1, train_data_2], axis=0)
    train_labels = np.concatenate([train_labels_1, train_labels_2], axis=0)
    train_rec = np.concatenate([train_rec_1, train_rec_2 + len(np.unique(train_rec_1))], axis=0)
    
    # Process session 3 for validation and test
    print("\nProcessing validation/test data...")
    val_test_data, val_test_labels, val_test_rec = process_session_data(session_id=3)
    
    # Split session 3 data between validation and test
    unique_recordings = np.unique(val_test_rec)
    n_val_recordings = int(len(unique_recordings) * val_ratio)
    
    # Randomly shuffle recordings
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(unique_recordings)
    
    val_recordings = unique_recordings[:n_val_recordings]
    test_recordings = unique_recordings[n_val_recordings:]
    
    # Split data based on recordings
    val_mask = np.isin(val_test_rec, val_recordings)
    val_data = val_test_data[val_mask]
    val_labels = val_test_labels[val_mask]
    
    test_mask = np.isin(val_test_rec, test_recordings)
    test_data = val_test_data[test_mask]
    test_labels = val_test_labels[test_mask]
    
    # Shuffle training data by recordings
    unique_train_recordings = np.unique(train_rec)
    np.random.shuffle(unique_train_recordings)
    train_order = np.concatenate([np.where(train_rec == rec)[0] for rec in unique_train_recordings])
    train_data = train_data[train_order]
    train_labels = train_labels[train_order]
    
    # Print shapes
    print("\nFinal shapes:")
    print(f"Train data: {train_data.shape}, labels: {train_labels.shape}")
    print(f"Validation data: {val_data.shape}, labels: {val_labels.shape}")
    print(f"Test data: {test_data.shape}, labels: {test_labels.shape}")
    
    # Save arrays
    print("\nSaving arrays...")
    np.save(os.path.join(save_dir, 'train.npy'), train_data)
    np.save(os.path.join(save_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(save_dir, 'validation.npy'), val_data)
    np.save(os.path.join(save_dir, 'validation_labels.npy'), val_labels)
    np.save(os.path.join(save_dir, 'test.npy'), test_data)
    np.save(os.path.join(save_dir, 'test_labels.npy'), test_labels)
    
    # Save metadata
    metadata = {
        'train_shape': train_data.shape,
        'val_shape': val_data.shape,
        'test_shape': test_data.shape,
        'n_channels': train_data.shape[1],
        'timesteps_per_segment': train_data.shape[2],
        'label_meanings': {
            0: 'neutral',
            1: 'sad',
            2: 'fear',
            3: 'happy'
        }
    }
    np.save(os.path.join(save_dir, 'metadata.npy'), metadata)
    
    return metadata

# Run the processing
metadata = save_train_val_test_split(save_dir='seed_iv/session')

# Print label distribution
def print_label_distribution(labels, split_name):
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n{split_name} label distribution:")
    for label, count in zip(unique, counts):
        print(f"Label {label}: {count} samples ({count/len(labels)*100:.1f}%)")

# Load and print distributions
train_labels = np.load('seed_iv/session/train_labels.npy')
val_labels = np.load('seed_iv/session/validation_labels.npy')
test_labels = np.load('seed_iv/session/test_labels.npy')

print_label_distribution(train_labels, "Training")
print_label_distribution(val_labels, "Validation")
print_label_distribution(test_labels, "Test")