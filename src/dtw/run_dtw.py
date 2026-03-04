import sys
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import glob
import subsequence_dtw_functions as dtw
from determine_ground_truth import calc_ground_truth, haversine_distance

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
        
        '''self.query_start = self.config['time']['query_start']
        self.query_length = self.config['time']['query_length']
        
        self.ref_start = self.config['time']['ref_start']
        self.ref_length = self.config['time']['ref_length']'

        print(f"\nTime windows (seconds from traverse start):")
        print(f"  Query: {self.query_start}, length: {self.query_length}s")
        print(f"  Reference: {self.ref_start}, length: {self.ref_length}s")'''

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
        
    def load_event_slice(self, folder, seq_start, seq_length):
        """
        Load events from batch files between start_offset and end_offset seconds
        from the beginning of the traverse
        """
        # Get the absolute start time of the traverse
        traverse_start_unix = self.get_first_file_time(folder)
        # Calculate absolute time window
        print(f"start_offset: {seq_start}")
        print(f"seq_length: {seq_length}")

        abs_seq_start_time = traverse_start_unix + seq_start
        abs_seq_end_time = abs_seq_start_time + seq_length
        
        print(f"\n  Traverse start (UNIX): {traverse_start_unix:.6f}")
        print(f"  Traverse start (Brisbane): {self.unix_to_brisbane(traverse_start_unix)}")
        print(f"  Window: {seq_start}s to {seq_start + seq_length}s from start")
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
                window_events = window_events[:,:4]
                window_events[:, 2] = window_events[:, 2].astype(int)
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
        events = np.vstack(all_events)
        # Sort by timestamp (just in case)
        events_ts = events[events[:, 3].argsort()]
        events_dtw = events_ts[:, :3]
        print("\n  Sample of sorted events (first 3):")
        print("  [x, y, polarity, timestamp]")
        for event in events[:10]:
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
        
        return events_dtw, events_ts
    
    
    def run_dtw(self, query_events, ref_events, ref_events_ts, pair):
        """Run subsequence DTW for a single pair"""
        print(f"\nQuery data shape: {query_events.shape}")
        print(f"Reference data shape: {ref_events.shape}")
        
        # Run subsequence DTW
        print("\nComputing DTW...")
        C, _, P = dtw.subsequence_dtw(query_events, ref_events, print_en=1)
        
        # Extract results
        a_ast = P[0, 1]      # Start index in reference
        b_ast = P[-1, 1]      # End index in reference
        
        # Get corresponding timestamps
        ref_start_time = ref_events_ts[a_ast, 3]
        ref_end_time = ref_events_ts[b_ast, 3]
        
        print("\n" + "="*40)
        print(f"DTW RESULTS - Pair {pair['pair_id']}")
        print("="*40)
        print(f"Type: {pair['type']}")
        print(f"Expected: {pair['expected_result']}")
        print(f"\nMATCHED SEGMENT:")
        print(f"  Start index: {a_ast} → UNIX: {ref_start_time:.6f}")
        print(f"  End index: {b_ast} → UNIX: {ref_end_time:.6f}")
        print(f"  Brisbane: {self.unix_to_brisbane(ref_start_time)} to {self.unix_to_brisbane(ref_end_time)}")
        print(f"Final cost: {C[-1, b_ast]:.2f}")
        print(f"Warping path length: {len(P)}")
        
        return {
            'pair_id': pair['pair_id'],
            'type': pair['type'],
            'expected_result': pair['expected_result'],
            'a_ast': a_ast,
            'b_ast': b_ast,
            'ref_start_time': ref_start_time,
            'ref_end_time': ref_end_time,
            'final_cost': float(C[-1, b_ast])
        }
    
    # ADD this new execute() method:
    def execute(self):
        """Main execution method - process all pairs"""
        print("\n" + "="*60)
        print("PROCESSING ALL QUERY-REFERENCE PAIRS")
        print("="*60)
        
        all_results = []
        
        for pair in self.config['pairs']:  # Loop through each pair
            print("\n" + "="*60)
            print("="*60)
            
            # Load query events using pair-specific times
            print("\n--- LOADING QUERY EVENTS ---")
            query_events, _ = self.load_event_slice(
                self.query_folder, 
                pair['time']['query_start'],    # Use pair-specific value
                pair['time']['query_length']     # Use pair-specific value
            )
            
            # Load reference events using pair-specific times
            print("\n--- LOADING REFERENCE EVENTS ---")
            ref_events, ref_events_ts = self.load_event_slice(
                self.ref_folder, 
                pair['time']['ref_start'],       # Use pair-specific value
                pair['time']['ref_length']        # Use pair-specific value
            )
            
            # Run DTW for this pair
            result = self.run_dtw(query_events, ref_events, ref_events_ts, pair)
            all_results.append(result)
        
        return all_results
    
    
       


# Simple function to save results if needed
def save_results(results, output_file="dtw_results.json"):
    """Save DTW results to file"""
    import pickle
    import pandas as pd
    
    # Save full results as JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Also save as CSV for easy viewing
    csv_file = output_file.replace('.json', '.csv')
    pd.DataFrame(results).to_csv(csv_file, index=False)
    
    print(f"\nResults saved to {output_file} and {csv_file}")
        

###Main

if __name__ == "__main__":
    config_file = "dtw_config.json"
    if not os.path.exists(config_file):
        print(f"ERROR: Config file '{config_file}' not found!")
        sys.exit(1)

    dtw_exp = DTWFromConfig(config_file)
    results = dtw_exp.execute()
    
    # Optional: Save results
    save_results(results)
        



