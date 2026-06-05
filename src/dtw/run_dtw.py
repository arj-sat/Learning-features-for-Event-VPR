import sys
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import glob
import subsequence_dtw_functions as dtw
from determine_ground_truth import calc_ground_truth
import time

class DTWFromConfig:
    def __init__(self, config_file):
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        print("="*60)
        print(f"DTW Experiment: {self.config['experiment_name']}")
        print("="*60)

        self.query_folder = self.config['data']['query_folder']
        self.ref_folder = self.config['data']['ref_folder']
        self.validation_threshold = self.config.get('validation_threshold', 50)
        
        '''self.query_start = self.config['time']['query_start']
        self.query_length = self.config['time']['query_length']
        
        self.ref_start = self.config['time']['ref_start']
        self.ref_length = self.config['time']['ref_length']'

        print(f"\nTime windows (seconds from traverse start):")
        print(f"  Query: {self.query_start}, length: {self.query_length}s")
        print(f"  Reference: {self.ref_start}, length: {self.ref_length}s")'''

    def density_select(self , events_ts, events_per_window=10, window_size_sec=0.1):
        """
        Subsample events to keep only top N events per time window based on density_value
    
        Args:
            events_ts: Array of events with columns [x, y, polarity, timestamp, density_value]
                    (the full events with timestamps and density)
            events_per_window: Number of events to keep per time window (default: 10)
            window_size_sec: Size of time window in seconds (default: 0.1)
        
        Returns:
            events_dtw: Subsampled events with columns [x, y, polarity] (for DTW)
            events_ts: Subsampled events with all columns (for timestamp reference)
        """
        if not use_topk:
            print('Not top k')
            return events_ts[:,:3], events_ts
        print(events_ts.shape)
        print(events_ts[0])
        print(events_ts[1])
        sort_indices = np.argsort(events_ts[:, 3])
        events_ts_sorted = events_ts[sort_indices]

        min_time = events_ts_sorted[0, 3]
        max_time = events_ts_sorted[-1, 3]
        num_windows = int(np.ceil((max_time - min_time) / window_size_sec))
        print(f"\n  Uniformising events:")
        print(f"    Original events: {len(events_ts_sorted)}")
        print(f"    Time range: {min_time:.3f} to {max_time:.3f}s")
        print(f"    Windows: {num_windows} windows of {window_size_sec}s")
        print(f"    Keeping top {events_per_window} events/window")

        selected_indices = []
        for i in range(num_windows):
            window_start = min_time + (i * window_size_sec)
            window_end = window_start + window_size_sec
            
            # Find events in this window
            mask = (events_ts_sorted[:, 3] >= window_start) & (events_ts_sorted[:, 3] < window_end)
            window_events_indices = np.where(mask)[0]

            if len(window_events_indices) > 0:
            # Get density values for events in this window
                density_values = events_ts_sorted[window_events_indices, 4]
                
                # Get indices of top N events (highest density values)
                if len(window_events_indices) <= events_per_window:
                    # Keep all events in this window
                    selected_indices.extend(window_events_indices)
                else:
                    # Keep only top N events
                    top_indices_local = np.argsort(density_values)[-events_per_window:]
                    top_global_indices = window_events_indices[top_indices_local]
                    selected_indices.extend(top_global_indices)

        selected_indices = sorted(selected_indices)

        # Create subsampled arrays
        events_ts_subsampled = events_ts_sorted[selected_indices]
        events_dtw_subsampled = events_ts_subsampled[:, :3]
        events_dtw_subsampled[:, 2] = events_dtw_subsampled[:, 2].astype(int)
        
        print(f"    After subsampling: {len(events_ts_subsampled)} events")
        print(f"    Reduction: {(1 - len(events_ts_subsampled)/len(events_ts_sorted)) * 100:.1f}%")
        
        return events_dtw_subsampled, events_ts_subsampled

    def unix_to_brisbane(self, unix_time):
        """Convert UNIX timestamp to Brisbane time (UTC+10)"""
        # Brisbane is UTC+10 (no daylight saving)
        brisbane_time = datetime.fromtimestamp(unix_time, tz=timezone.utc) + pd.Timedelta(hours=10)
        return brisbane_time.strftime('%Y-%m-%d %H:%M:%S')
        
    def get_first_file_time(self, folder):
        """Get the timestamp of the first event in the first batch file"""
        # Find all CSV files in folder
        csv_files = sorted(glob.glob(os.path.join(folder, "filtered_events_batch_*.csv")))
        
        if not csv_files:
            raise FileNotFoundError(f"No batch files found in {folder}")
        
        # Read first file
        first_file = csv_files[0]
        print(f"  Reading first file: {os.path.basename(first_file)}")
        
        # Read first row to get start time
        df1 = pd.read_csv(first_file, nrows=2, header= None)
        first_timestamp = df1.iloc[1, 3] 
        #print("First timestamp " + first_timestamp)
        return float(first_timestamp)
        
    def load_event_slice(self, folder, seq_start, seq_end):
        """
        Load events from batch files between start_offset and end_offset seconds
        from the beginning of the traverse
        """
        # Get the absolute start time of the traverse
        traverse_start_unix = self.get_first_file_time(folder) 
        # Calculate absolute time window
        print(f"start_offset: {seq_start}")
        print(f"seq_length: {seq_end - seq_start}")

        #abs_seq_start_time = traverse_start_unix + seq_start
        abs_seq_start_time = seq_start
        #abs_seq_end_time = abs_seq_start_time + seq_length
        abs_seq_end_time = seq_end
        
        print(f"\n  Traverse start (UNIX): {traverse_start_unix:.6f}")
        print(f"  Traverse start (Brisbane): {self.unix_to_brisbane(traverse_start_unix)}")
        print(f"  Absolute: {abs_seq_start_time:.6f} to {abs_seq_end_time:.6f} UNIX")
        print(f"  Brisbane time: {self.unix_to_brisbane(abs_seq_start_time)} to {self.unix_to_brisbane(abs_seq_end_time)}")
        
        # Find all batch files
        print(folder)
        csv_files = sorted(glob.glob(os.path.join(folder, "filtered_events_batch_*.csv")))
        print(f"  Found {len(csv_files)} batch files")
        
        all_events = []
        total_events = 0
        files_processed = 0
        stop = False
        files_with_events = [] 
        
        # Iterate through files and collect events in time window
        for csv_file in csv_files:
            files_processed +=1
            #print(f"\n    Reading file: {os.path.basename(csv_file)}")
            df = pd.read_csv(csv_file, header=None, skiprows=1)
            timestamps = df.iloc[:, 3].values
                
            if np.any((timestamps >= abs_seq_start_time) & (timestamps <= abs_seq_end_time)):
                # Filter events in the window
                mask = (timestamps >= abs_seq_start_time) & (timestamps <= abs_seq_end_time)
                window_events =df[mask].values
                window_events = window_events[:,:5]
                window_events[:, 2] = window_events[:, 2].astype(int) # converting polarity to integers
                if len(window_events) > 0:
                    all_events.append(window_events)
                    #file_events_count += len(window_events)
                    files_with_events.append((os.path.basename(csv_file), len(window_events)))
                    total_events += len(window_events)
                
                # If we've passed the end time, we can stop
                if np.min(timestamps) > abs_seq_end_time:
                    stop = True
                    break
            
            if(stop):
                print(f"Last file processed {csv_file}for this event slice")
                break


        
        for file_name, count in files_with_events:
            print(f"    - {file_name}: {count} events")
        
        # Combine all events
        if len(all_events) == 0:
            return np.array([]), np.array([])
        events = np.vstack(all_events)
        # Sort by timestamp (just in case)
        events_ts = events[events[:, 3].argsort()]
        #events_dtw = events_ts[:, :3]
        print("\n  Sample of sorted events (first 3):")
        print("  [x, y, polarity, timestamp]")
        for event in events_ts[:5]:
            print(f"  {event}")
        
        # Normalize time to start at 0 for DTW (as done in the original code)
        '''first_timestamp = events[0, 3]
        events_normalized = events.copy()

        #events_normalized[:, 3] -= first_timestamp
        print("Printing sample of normalised events")
        for event in events_normalized[:11]:
            print(f"  {event}")
        print("")
        print(f"  Collected {total_events} events from {files_processed} files")
        print(f"  Normalized time range: {events_normalized[0,3]:.3f} - {events_normalized[-1,3]:.3f}s")'''
        
        return events_ts
    
    
    def run_dtw(self, query_events, ref_events, ref_events_ts, pair,events_per_window):
        """Run subsequence DTW for a single pair"""
        
        # Run subsequence DTW
        print("\nComputing DTW...")
        s_time = time.time()
        C, _, P = dtw.subsequence_dtw(query_events, ref_events, print_en=1)
        
        # Extract results
        a_ast = P[0, 1]      # Start index in reference
        b_ast = P[-1, 1]      # End index in reference
        
        # Get corresponding timestamps
        ref_start_time =  self.unix_to_brisbane(ref_events_ts[a_ast, 3])
        ref_end_time = self.unix_to_brisbane(ref_events_ts[b_ast, 3])

        #Check match
        matched_ref_ts = ref_events_ts[(a_ast + b_ast) // 2, 3]
        expected_match_ts = pair['ref_match']
        timestamp_error = abs(matched_ref_ts - expected_match_ts)
        timestamp_match = timestamp_error < 5  # Within 1 second
        
        print("\n" + "="*40)
        print(f"DTW RESULTS - Pair {pair['pair_id']}")
        print("="*40)
        print(f"Type: {pair['type']}")
        print(f"\nQuery data shape: {query_events.shape}")
        print(f"Reference data shape: {ref_events.shape}")

        print(f"\nMATCHED SEGMENT:")
        print()
        print(f"Start index: {a_ast} → UNIX: {ref_start_time}")
        print(f"End index: {b_ast} → UNIX: {ref_end_time}")
        print(f"Expected match timestamp: {self.unix_to_brisbane(expected_match_ts)}")
        print(f"Actual matched timestamp: {self.unix_to_brisbane(matched_ref_ts)}")
        print(f"Timestamp error: {timestamp_error:.3f} seconds")
        print(f"Timestamp matches expected: {'✓ YES' if timestamp_match else '✗ NO'}")
        print(f"Final cost: {C[-1, b_ast]}")
        print(f"Warping path length: {len(P)}")
        e_time = time.time()
        print(f"time taken for pair: {round((e_time - s_time)/60, 2)} min")

        result =  {
            'pair_id': pair['pair_id'],
            'events_per_window': events_per_window,
            'type': pair['type'],  
            'a_ast': a_ast,
            'b_ast': b_ast, 
            'query_start_time_ts': pair["query_start"],
            'query_start_time': self.unix_to_brisbane(pair["query_start"]),
            'query_len': pair["query_l"],
            'query_events': query_events.shape ,
            'ref_start_time_ts':pair["ref_start"],
            'ref_start_time': self.unix_to_brisbane(pair["ref_start"]),
            'ref_len': pair["ref_l"],
            'ref_events':  ref_events.shape ,
            #'matched_ref_start':ref_start_time,
            #'matched_ref_end': ref_end_time,
            'matched_ref_timestamp': matched_ref_ts,
            'expected_ref_timestamp': expected_match_ts,
            'timestamp_error_seconds': timestamp_error,
            'timestamp_match': timestamp_match,
            'final_cost': float(C[-1, b_ast]),
            'time_taken_seconds': e_time - s_time
        }

        return result
    
    # ADD this new execute() method:
    def execute(self,pair_ids=None):
        """Main execution method - process all pairs"""
        print("\n" + "="*60)
        print("PROCESSING ALL QUERY-REFERENCE PAIRS")
        print("="*60)
        
        all_results = []
        
        events_per_window = self.config.get('events_per_0_1sec', 25)
        print(f"events per window :{events_per_window}")
        window_size = self.config.get('window_size_sec', 0.1)
        max_noof_pairs = 100
        z = 0
        for pair in self.config['pairs']:  # Loop through each pair
            if pair_ids is not None and pair['pair_id'] not in pair_ids:
                continue
            print("\n" + "="*60)
            print("="*60)
            z = z + 1

            # Load query events using pair-specific times
            print("\n--- LOADING QUERY EVENTS ---")
            query_events_ts = self.load_event_slice(
                self.query_folder, 
                pair['query_start'],    # Use pair-specific value
                pair['query_end']     # Use pair-specific value
            )
            print(query_events_ts.shape)
            # Load reference events using pair-specific times
            print("\n--- LOADING REFERENCE EVENTS ---")
            ref_events_ts = self.load_event_slice(
                self.ref_folder, 
                pair['ref_start'],       # Use pair-specific value
                pair['ref_end']        # Use pair-specific value
            )
                
            # Choosing top events
            print("\n--- PICKING TOP EVENTS ---")
            query_events, query_events_ts = self.density_select(
                query_events_ts, 
                events_per_window=events_per_window,
                window_size_sec=window_size
            )
            print(f"Query events shape after top {events_per_window}: {query_events.shape}")

            ref_events, ref_events_ts = self.density_select(
                ref_events_ts,
                events_per_window=events_per_window,
                window_size_sec=window_size
            )
            print(f"Reference events shape after top {events_per_window}: {ref_events.shape}")
            #exit()
            
            # Run DTW for this pair
            result = self.run_dtw(query_events, ref_events, ref_events_ts, pair,events_per_window)
            all_results.append(result)
            if(z == max_noof_pairs):
                break
        
        return all_results
    
    def validate_with_gps(self, query_name, query_end_time, ref_name, ref_end_time):
        """
        Validate if the matched segment corresponds to the same physical location using GPS
        
        Args:
            query_name: Name of query dataset (e.g., 'sunset1', 'night', etc.)
            query_end_time: End time of query segment (seconds from start)
            ref_name: Name of reference dataset
            ref_end_time: End time of matched segment in reference (seconds from start)
        
        Returns:
            dict: Validation results including distance and pass/fail status
        """
        print(f"\n--- GPS VALIDATION ---")
        print(f"Query: {query_name} @ {query_end_time:.2f}s")
        print(f"Reference: {ref_name} @ {ref_end_time:.2f}s")
        
        try:
            reference_path, query_position, estimated_position, distance, groundtruth_index = calc_ground_truth(
                query_name, query_end_time, ref_name, ref_end_time
            )
            
            # Determine if match is valid (within threshold)
            is_valid = distance <= self.validation_threshold
            
            print(f"\nGPS Validation Results:")
            print(f"  Query position (lat, lon): ({query_position[0]:.6f}, {query_position[1]:.6f})")
            print(f"  Estimated position (lat, lon): ({estimated_position[0]:.6f}, {estimated_position[1]:.6f})")
            print(f"  Distance between positions: {distance:.2f} meters")
            print(f"  Ground truth index in reference: {groundtruth_index}")
            print(f"  Validation threshold: {self.validation_threshold}m")
            print(f"  Match valid: {'✓ YES' if is_valid else '✗ NO'}")
            
            return {
                'query_position': query_position.tolist() if hasattr(query_position, 'tolist') else query_position,
                'estimated_position': estimated_position.tolist() if hasattr(estimated_position, 'tolist') else estimated_position,
                'distance_meters': float(distance),
                'groundtruth_index': int(groundtruth_index),
                'is_valid': bool(is_valid),
                'threshold_meters': self.validation_threshold
            }
            
        except Exception as e:
            print(f"ERROR in GPS validation: {e}")
            return {
                'error': str(e),
                'is_valid': False,
                'distance_meters': None
            }
       


# Simple function to save results if needed
def save_results(results, exp_no):
    import os   
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    df.to_csv(RUN_PATH +  f"/{exp_no}.csv", index=False)
    
    print(f"Columns: {', '.join(df.columns.tolist())}")
    print(f"\nResults saved as csv")

        

###Main

if __name__ == "__main__":
    exp_no = "0_longer" 
    pair_ids_to_run = None#[2, 3, 4,8,12,14] 
    global RUN_PATH
    RUN_PATH = r"C:\Arjun\Thesis\results\exp_0\longer"
    global use_topk 
    use_topk =  True
    #config_file = f"{exp_no}_mdtw_config.json"  # no leading slash
    config_file = "0_longer_mdtw_config.json"
    config_path = os.path.join(RUN_PATH, config_file)  # let os handle separators

    print(config_path)

    if not os.path.exists(config_path):
        print(f"ERROR: Config file '{config_path}' not found!")
        sys.exit(1)

    dtw_exp = DTWFromConfig(config_path)
    results = dtw_exp.execute(pair_ids_to_run)
    save_results(results, exp_no=exp_no)
            



