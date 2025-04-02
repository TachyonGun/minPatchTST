import os
import numpy as np
import pandas as pd
from glob import glob

def clean_column_name(name):
    """Clean column name to remove problematic characters"""
    # Replace unknown/invalid characters with underscore
    cleaned = ''.join(char if char.isprintable() and ord(char) < 128 else '_' for char in str(name))
    return cleaned

def process_dataset_folder(folder_name):
    """
    Process all CSV files in a dataset folder and combine them into a single numpy array
    Also returns the column names from the first CSV file
    """
    base_path = os.path.join('all_six_datasets', folder_name)
    
    # Get all CSV files in the folder
    csv_files = glob(os.path.join(base_path, '*.csv'))
    if not csv_files:
        print(f"No CSV files found in {base_path}")
        return None, None
    
    # Read and combine all CSV files
    all_data = []
    column_names = None  # Store column names from first file
    
    for i, csv_file in enumerate(csv_files):
        print(f"Processing {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Remove date/time columns if they exist
        # Common date column names in time series data
        date_columns = ['date', 'time', 'timestamp', 'datetime', 'Date', 'Time']
        for col in date_columns:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Store column names from first file after removing date columns
        if i == 0:
            # Clean column names to handle Unicode characters
            column_names = [clean_column_name(col) for col in df.columns]
            # Rename columns in dataframe to match cleaned names
            df.columns = column_names
        else:
            # Clean current file's column names
            current_columns = [clean_column_name(col) for col in df.columns]
            # Verify column names match across files
            if current_columns != column_names:
                print(f"Warning: Column names in {csv_file} don't match the first file")
                print(f"Expected: {column_names}")
                print(f"Got: {current_columns}")
                return None, None
            # Rename columns in dataframe to match cleaned names
            df.columns = column_names
        
        # Convert to numpy array and ensure float32 type
        data = df.values.astype(np.float32)
        all_data.append(data)
    
    # Concatenate all arrays if multiple files exist
    if len(all_data) > 1:
        combined_data = np.concatenate(all_data, axis=0)
    else:
        combined_data = all_data[0]
    
    return combined_data, column_names

def main():
    # List of dataset folders to process
    datasets = ['electricity', 'ETT-small', 'exchange_rate', 'illness', 'traffic', 'weather']
    
    # Process each dataset
    for dataset in datasets:
        print(f"\nProcessing {dataset} dataset...")
        data, column_names = process_dataset_folder(dataset)
        
        if data is not None and column_names is not None:
            # Save the combined numpy array
            output_path = os.path.join('all_six_datasets', f'{dataset}.npy')
            np.save(output_path, data)
            
            # Save column names to a text file with UTF-8 encoding
            column_names_path = os.path.join('all_six_datasets', f'{dataset}_columns.txt')
            with open(column_names_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(column_names))
            
            print(f"Saved {dataset} dataset with shape {data.shape} to {output_path}")
            print(f"Saved {len(column_names)} column names to {column_names_path}")
        else:
            print(f"Failed to process {dataset} dataset")

if __name__ == "__main__":
    main() 