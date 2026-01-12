from sampler import FilterDataRecursive
from torch_geometric.data import Data
import time
import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path



def csv_to_pyg_data(df: pd.DataFrame, 
                   x_col: str = 'x',
                   y_col: str = 'y', 
                   p_col: str = 'polarity',
                   t_col: str = 'timestamp') -> Data:

    
    # Print column info for debugging
    print(f"CSV columns: {list(df.columns)}")
    print(f"First few rows:\n{df.head()}")

    x_coords = df[x_col].values.astype(np.float32)
    y_coords = df[y_col].values.astype(np.float32)
    timestamps = df[t_col].values.astype(np.float32)
    polarities = df[p_col].values

    polarities = df[p_col].values.astype(np.float32) * 2 - 1    
    pos = np.column_stack([x_coords, y_coords, timestamps])
    pos = torch.from_numpy(pos)
    
    # Create polarity tensor
    x = torch.from_numpy(polarities).reshape(-1, 1)
    
    # Create Data object
    data = Data(x=x, pos=pos)
    
    return data
    
def process_csv_file(csv_path: str, filter_instance: FilterDataRecursive, 
                    sampling_threshold: float = 0.01) -> pd.DataFrame:
    """Process a single CSV file and return filtered events."""
        
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} events")
    
    # Convert to PyG Data
    data = csv_to_pyg_data(df)
    
    # Get filter values
    print("  Computing density values...")
    filter_values = filter_instance.subsample(data)
    
    # Apply threshold-based sampling
    print("  Applying threshold-based sampling...")
    np.random.seed(42)
    random_values = np.random.rand(len(filter_values))
    keep_mask = random_values < (sampling_threshold * filter_values)
    
    # Filter events
    filtered_df = df[keep_mask].copy()
    filtered_df['density_value'] = filter_values[keep_mask]
    
    print(f"  Kept {len(filtered_df)} events ({len(filtered_df)/len(df)*100:.1f}%)")
    with open(LOG_FILE, 'a') as f:
        print(f"  Kept {len(filtered_df)} events ({len(filtered_df)/len(df)*100:.1f}%)")

    return filtered_df


def main():
    
    #os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    INPUT_FOLDER = r"C:\Arjun\Thesis\data\New folder\filtered chunks"
    OUTPUT_FOLDER = r"C:\Arjun\Thesis\data\New folder\filtered chunks\subsampled"
    global LOG_FILE
    LOG_FILE = r"C:\Arjun\Thesis\data\New folder\filtered chunks\subsampled\process_log.txt"


    '''
    spatiotemporal_filtering_subsampling: #Subsampling the events randomly with probability proportional to spatiotemporal filtering values
      transform: null #True
      tau: null #30 in milliseconds. Temporal constant for spatiotemporal filtering
      filter_size: null #7 in pixels. Filter size for spatial Gaussian filter.
      sampling_threshold: null #0.1 The smaller te threshold, the smaller number of events
      normalization_length: null #null #The length of the filter_values normalization, e.g., 100 (If None, the normalization is not applied)
      mean_normalized: null #False If True, the filter values are normalized by dividing them by their mean
    '''

    IMAGE_SIZE = (346, 260)  # (height, width) - ADJUST THIS!
    
    # Filter parameters
    TAU_MS = 30.0  # Temporal decay in milliseconds
    FILTER_SIZE = 7  # Spatial filter size (must be odd)
    
    # Sampling parameters
    SAMPLING_THRESHOLD = 0.05  #0.01, 0.1

    #data = csv_to_pyg_data(CSV_FILE)

    print("\n2. Initializing causal density filter...")
    print(f"   Image size: {IMAGE_SIZE}")
    print(f"   Tau: {TAU_MS} ms")
    print(f"   Filter size: {FILTER_SIZE}")
    
    filter_instance = FilterDataRecursive(TAU_MS, FILTER_SIZE, IMAGE_SIZE)

    csv_files = list(Path(INPUT_FOLDER).glob("*.csv"))
    print(f"\nFound {len(csv_files)} CSV files to process")
    with open(LOG_FILE, 'w') as f:
        f.write("=== CSV Processing Log ===\n")
        f.write(f"\nFound {len(csv_files)} CSV files to process")
        

    csv_files_sorted = sorted(csv_files, key=lambda x: int(x.stem.split('_')[-1]))

    '''print("\nFirst 5 CSV files (in processing order):")
    for i, csv_file in enumerate(csv_files_sorted[:5]):
        print(f"  {i+1}. {csv_file.name}")
    
    print("\nLast 5 CSV files (in processing order):")
    for i, csv_file in enumerate(csv_files_sorted[-5:], start=len(csv_files_sorted)-4):
        print(f"  {i+1}. {csv_file.name}")'''

    for csv_file in csv_files_sorted:
        print(f"Loading {csv_file.name}")
        with open(LOG_FILE, 'a') as f:
            f.write(f"Loading {csv_file.name}\n")
        # Process the file
        filtered_df = process_csv_file(str(csv_file), filter_instance, SAMPLING_THRESHOLD)
        
        # Save filtered results
        output_path = os.path.join(OUTPUT_FOLDER, f"filtered_{csv_file.name}")
        filtered_df.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")



if __name__ == '__main__':
    stime = time.time()
    main()
    total_time_minutes = (time.time() - stime) / 60.0
    print(f"\n  Total time for code completetion: {total_time_minutes:.2f} minutes")