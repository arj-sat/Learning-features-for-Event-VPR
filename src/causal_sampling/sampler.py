
from scipy.ndimage import gaussian_filter
import numpy as np
import time
from typing import Union, Iterable, List, Tuple
from torch_geometric.data import Data, HeteroData

class FilterDataRecursive():

    def __init__(self, tau: float, filter_size: int, image_size: Tuple[int,int]):
        
        assert filter_size % 2 == 1, "Filter size must be odd"
        self.tau = tau
        self.filter_size = filter_size
        self.image_size = image_size
        self.K = filter_size // 2
        self.W, self.H = image_size
   
        
        sigma = filter_size / 5.0
        kernel = np.zeros((filter_size, filter_size))
        kernel[filter_size // 2, filter_size // 2] = 1
        self.gaussian_kernel = gaussian_filter(kernel, sigma)
        self.gaussian_kernel = self.gaussian_kernel / np.sum(self.gaussian_kernel)

    def subsample(self, data):

        '''print("\n=== DEBUG: Checking Data Structure ===")
        print(f"data.pos shape: {data.pos.shape}")
        print(f"data.x shape: {data.x.shape}")
        
        # Print first 5 events with all indexing methods
        print("\nFirst 5 events:")
        for i in range(min(5, len(data.pos))):
            ts = data.pos[i]
            print(f"Event {i}:")
            print(f"  Raw ts: {ts}")
            print(f"  ts[0], ts[1], ts[2]: {ts[0].item():.0f}, {ts[1].item():.0f}, {ts[2].item():.2e}")
            print(f"  ts[-3], ts[-2], ts[-1]: {ts[-3].item():.0f}, {ts[-2].item():.0f}, {ts[-1].item():.2e}")
            print(f"  x[i] (polarity): {data.x[i].item()}")
            print()
        
        # Exit after printing debug info
        print("Exiting after debug print...")
        exit()''' 

        self.tau = int(self.tau * 1000)
        self.last_time_tensor = np.full((2,self.H,self.W), float('0') , dtype=np.float32)
        print(f"Image width: {self.image_size[0]}, Image height: {self.image_size[1]}")
        self.temporal_accumulation_tensor = np.full((2,self.image_size[1],self.image_size[0]), float('0') , dtype=np.float32)

        filter_value_recursive = np.zeros(data.pos.shape[0], dtype=np.float32)

        start_time = time.time()
        for i ,ts in enumerate(data.pos):
    
            pp = 0 if data.x[i] < 0 else 1
    
            h = ts[-2].int()
            w = ts[-3].int()
            t = ts[-1].numpy()

            if i % 100000 == 0:
                print(f"  Event {i}: Height: {h}, Width: {w}")
    
            h_start = max(h - self.K, 0)
            h_end = min(h + self.K, self.H-1)
            w_start = max(w - self.K, 0)
            w_end = min(w + self.K, self.W-1)

            
            # Compute the temporal lag
            #print(f"tau: {self.tau}")
            temporal_lag = np.exp(- (t - self.last_time_tensor[pp,h_start:h_end+1,w_start:w_end+1])/self.tau)

            # update the last time tensor
            self.last_time_tensor[pp,h_start:h_end+1,w_start:w_end+1] = t

            # update the temporal accumulation tensor

            self.temporal_accumulation_tensor[pp,h_start:h_end+1,w_start:w_end+1] *= temporal_lag
            self.temporal_accumulation_tensor[pp,h,w] += 1

            # Compute the filter value
            filter_value_recursive[i] = np.sum(self.temporal_accumulation_tensor[pp,h_start:h_end+1,w_start:w_end+1] * self.gaussian_kernel[h_start - h + self.K:h_end + 1 - h + self.K, w_start - w + self.K:w_end +1 - w + self.K])
    
        end_time = time.time()
        total_time_minutes = (end_time - start_time) / 60.0
        print(f"\n  Total processing time: {total_time_minutes:.2f} minutes")
        
        return filter_value_recursive