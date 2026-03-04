import numpy as np
from tqdm import tqdm
import libfmp.c3
import scipy
import matplotlib.pyplot as plt

def compute_accumulated_cost_matrix_subsequence_dtw(C):
    """Given the cost matrix, compute the accumulated cost matrix for
    subsequence dynamic time warping with step sizes {(1, 0), (0, 1), (1, 1)}
    Args:
        C (np.ndarray): Cost matrix
    Returns:
        D (np.ndarray): Accumulated cost matrix
    """
    # Create cost array
    N, M = C.shape
    D = np.zeros((N, M))
    D[:, 0] = np.cumsum(C[:, 0])
    D[0, :] = C[0, :]

    for n in tqdm(range(1, N), 'Calculating Accumulated Cost', leave=False):
        for m in range(1, M):
            D[n, m] = C[n, m] + min(D[n-1, m], D[n, m-1], D[n-1, m-1])

    return D


def compute_optimal_warping_path_subsequence_dtw(D, m=-1):
    """Given an accumulated cost matrix, compute the warping path for
    subsequence dynamic time warping with step sizes {(1, 0), (0, 1), (1, 1)}
    Args:
        D (np.ndarray): Accumulated cost matrix
        m (int): Index to start back tracking; if set to -1, optimal m is used (Default value = -1)
    Returns:
        P (np.ndarray): Optimal warping path (array of index pairs)
    """
    N, M = D.shape
    n = N - 1
    if m < 0:
        m = D[N - 1, :].argmin()
    P = [(n, m)]

    while n > 0:
        if m == 0:
            cell = (n - 1, 0)
        else:
            val = min(D[n-1, m-1], D[n-1, m], D[n, m-1])
            if val == D[n-1, m-1]:
                cell = (n-1, m-1)
            elif val == D[n-1, m]:
                cell = (n-1, m)
            else:
                cell = (n, m-1)
        P.append(cell)
        n, m = cell
    P.reverse()
    P = np.array(P)
    return P


def subsequence_dtw(query, reference, print_en=1):
    query = np.array(query, dtype=np.float64)
    reference = np.array(reference, dtype=np.float64)
    print(f'Shape: {query.shape}')
    print('First 3 rows:')
    print(query[:3])
    
    print('\nREFERENCE:')
    print(f'Shape: {reference.shape}')
    print('First 3 rows:')
    print(reference[:3])

    # Compute cost matrix using Euclidean distance
    if print_en: 
        print('Computing the cost matrix C:') 
    C =  libfmp.c3.compute_cost_matrix(query.T, reference.T, metric='euclidean')
    # Compute the accumulated cost
    if print_en:  
        print('Computing the accumulated cost matrix D:')
    D =  compute_accumulated_cost_matrix_subsequence_dtw(C)
    if print_en:
        print('Accumulated cost matrix complete \n')
    # Compute optimal warping path
    if print_en:
        print('Computing the optimal warping path:')
    P = compute_optimal_warping_path_subsequence_dtw(D)
    return C, D, P


def filter_data(data_array, filter_gap, cols=346, rows=260):
    if filter_gap == -1:
        return data_array
    else:
        x_filter = np.arange(0, cols+1, filter_gap)
        y_filter = np.arange(0, rows+1, filter_gap)
        x_idx = np.isin(data_array[:,1], x_filter)
        y_idx = np.isin(data_array[:,2], y_filter)
        idx = np.bitwise_and(x_idx, y_idx)
    return data_array[idx,:]


def select_data_sequence(data_array, start_time, end_time):
    ''' Function to slice the sequence time'''
    assert start_time < end_time, "The end time must proceed the start time."
    
    # check bounday conditions
    if (start_time < data_array[0,0]) or (start_time == -1):
        start_time = data_array[0,0]
    if end_time > data_array[-1,0] or (end_time == -1):
        end_time = data_array[-1,0]

    indexes = np.where((data_array[:,0] > start_time) & (data_array[:,0] < end_time))
    signal = data_array[indexes[0],:]
    return signal


def select_data(query, reference, selection_choice):
    assert selection_choice in (0, 1, 2, 3, 4, 5), 'Invalid Data Select mode: Must be 0, 1, 2, 3, 4, 5'
    # 0 - x, y      1 - x, y, sig       2 - x_norm, y_norm, sig     3 - dt, x, y, sig     4 - dt, x_norm, y_norm, sig      5 - dt, sig
    match selection_choice:
        case 0: #(x,y)
            query_data = query[:,1:3]
            reference_data = reference[:,1:3]
        case 1: # (x, y, sig)
            query_data = query[:,1:4]
            reference_data = reference[:,1:4]
        case 2: #(x_norm, y_norm, sig)
            query_xy_norm = query[:,1:3]/np.max(query[:,1:3], axis=0)
            reference_xy_norm = reference[:,1:3]/np.max(reference[:,1:3], axis=0)   
            query_data = np.zeros((query.shape[0], 3))
            query_data[:,0:2] = query_xy_norm
            query_data[:,2] = query[:,3]
            reference_data = np.zeros((reference.shape[0], 3))
            reference_data[:,0:2] = reference_xy_norm
            reference_data[:,2] = reference[:,3]
        case 3: #(dt, x, y, sig)
            query_data = np.zeros((query.shape[0]-1, 4))
            query_data[:,0] = np.diff(query[:,0])
            query_data[:,1:4] = query[1:,1:4]
            reference_data = np.zeros((reference.shape[0]-1, 4))
            reference_data[:,0] = np.diff(reference[:,0])
            reference_data[:,1:4] = reference[1:,1:4]
        case 4: #(dt, x_norm, y_norm, sig)
            query_xy_norm = query[:,1:3]/np.max(query[:,1:3], axis=0)
            reference_xy_norm = reference[:,1:3]/np.max(reference[:,1:3], axis=0)
            query_data = np.zeros((query.shape[0]-1, 4))
            query_data[:,0] = np.diff(query[:,0])
            query_data[:,1:3] = query_xy_norm[1:,:]
            query_data[:,3] = query[1:,3]
            reference_data = np.zeros((reference.shape[0]-1, 4))
            reference_data[:,0] = np.diff(reference[:,0])
            reference_data[:,1:3] = reference_xy_norm[1:,:]
            reference_data[:,3] = reference[1:,3]
        case 5: #dt, sig
            query_data = np.zeros((query.shape[0]-1, 2))
            reference_data = np.zeros((reference.shape[0]-1, 2))
            query_data[:,0] = np.diff(query[:,0])
            query_data[:,1] = query[1:, 3]
            reference_data[:,0] = np.diff(reference[:,0])
            reference_data[:,1] = reference[1:, 3]
        case other:
            # assertion at the beginning should avoid any case of this
            ValueError('Unknown Data Select Input, check value')
            
    return query_data, reference_data


def key_event_mask(query, reference, threshold=5,  rows = 260, cols = 346,):
    # Create mask for active pixel coordinates
    M = np.zeros((rows, cols))

    for i in range(len(query)):
        x, y = query[i,1:3]
        M[int(y),int(x)] += 1

    M[M<threshold] = 0
    M[M>=threshold] = 1

    # Filter the reference and the query
    filtered_reference = []

    for i in tqdm(range(len(reference)), 'Applying Mask to Reference:', leave=False):
        if M[int(reference[i,2]), int(reference[i,1])] == 1:
            filtered_reference.append(reference[i,:].T) 

    filtered_query = []
    for i in tqdm(range(len(query)), 'Applying Mask to Query:', leave=False):
        if M[int(query[i,2]), int(query[i,1])] == 1:
            filtered_query.append(query[i,:].T) 

    return np.asarray(filtered_reference), np.asarray(filtered_query), M


def remove_active_pixels(dataset, threshold=5,  rows = 260, cols = 346,):
    # Create mask for active pixel coordinates
    M = np.zeros((rows, cols))

    for i in range(len(dataset)):
        x, y = dataset[i,1:3]
        M[int(y),int(x)] += 1

    M[M > threshold] = 0
    # M[M != 0] = 1

    # Filter the reference and the query
    filtered_dataset = []

    for i in tqdm(range(len(dataset)), 'Applying Mask to Query:', leave=False):
        if M[int(dataset[i,2]), int(dataset[i,1])] != 0:
            filtered_dataset.append(dataset[i,:].T) 

    return np.asarray(filtered_dataset), M


def region_filter(dataset, threshold, resolution, dt=0.1, cols=346, rows=260, en_print=0):
    assert len(resolution)==2, 'resolution must be two dimensions with number of dimensions y and x --> [y, x] to match numpy array indexing'
    
    #---- Create data storage arrays ----#
    M_accumulator = np.zeros (resolution)
    M_prev_time = np.zeros (resolution)
    output_data = []

    #---- Determine the indexes based on the region size ----#
    y_indexes = np.linspace(-1,rows, resolution[0]+1 )
    x_indexes = np.linspace(-1,cols, resolution[1]+1 )

    #---- Process data ----#
    for i in tqdm(range(dataset.shape[0]), 'Performing data compression on the event dataset'):
    # for i in range(dataset.shape[0]):
        event = dataset[i,:]
        if event[3] == 1:
            pol = 1
        elif event[3] == 0:
            pol = -1
        else:
            raise ValueError('Polarity Error')
        
        # find the appropriate index in the downsampled array
        x_ind = np.searchsorted(x_indexes, event[1]) - 1 
        y_ind = np.searchsorted(y_indexes, event[2]) - 1    
        
        # update the accumulator value
        M_accumulator[y_ind, x_ind] += pol
        M_prev_time[y_ind, x_ind] = event[0]

        # check the accumulator value and release if necessary
        if abs(M_accumulator[y_ind, x_ind]) >= threshold:
            output_data.append([event[0], x_ind, y_ind, pol])
            if en_print:
                print(f"{event[0]} \t {x_ind} \t {y_ind} \t {pol} \t {i}")
            M_accumulator[y_ind, x_ind] = 0 

        # reset the cells that haven't been updated in a while 
        row_indices, col_indices = np.where(event[0]-M_prev_time > dt)

        if len(row_indices)>0 and len(col_indices)>0:
            # print(f'{row_indices} \t {col_indices}')
            M_accumulator[row_indices, col_indices] = 0 

    output_data = np.asarray(output_data)
   
    return output_data



def analyse_cost(cost_array, alpha, beta, window_size_param=0.1, window_shift=0.5, bound_size=50, show_diagnostics=0):
    '''
    analyse_cost takes an array of accumulated cost values and calculates metrics 
    to evaluate the confidence that we can have in the estimated position.
    Inputs:
        - cost_array: array of accumulated cost values. This is the final row of the
                      accumulated cost matrix.
        - query_length: the number of elements in the query signal. If this is provided,
                        the cost_array will be normalised by this value. 
    '''

    min_index = np.argmin(cost_array)
    # Apply Savgol filter
    y =  scipy.signal.savgol_filter(cost_array, len(cost_array)//100, 5)

    #---- Windowing to find next minimum values ----#
    window_param = 1/window_size_param
    window_size = int(len(y)//window_param)
    min_points = []

    start_index = 0
    i = 0

    while True:
        window_start = start_index
        window_end = window_start + window_size
        if window_end > len(y):
            break
        window_data = y[window_start:window_end]
        window_min_idx = np.argmin(window_data) # in window
        global_min_idx = window_min_idx + window_start # in full data
        window_min = np.min(window_data)

        if show_diagnostics:
            fig, ax = plt.subplots(1,2)
            ax[0].plot(window_data)
            ax[0].axvline(np.argmin(window_data), color='red')
            ax[1].plot(y)
            ax[1].axvline(global_min_idx, color='red')
            ax[1].axvline(window_start, color='red', ls='--')
            ax[1].axvline(window_end, color='red', ls='--')
            plt.show()
            print(f"Window Start: {window_start} \t Window End: {window_end} \t Min Index: {global_min_idx}")


        # Only consider indexes that are not are on the boundary as they may not be minimums
        if (window_min_idx != window_start) and (window_min_idx != window_end): 
            if window_start == 0:
                min_points.append([window_min, global_min_idx])
                if show_diagnostics:
                    print(f"{global_min_idx} added to list")
            else:
                if global_min_idx != min_points[-1][1]: #if the indexes are not the same
                    if (global_min_idx - min_points[-1][1] < bound_size):
                        if show_diagnostics:
                            print('Close points...')
                        if window_min < min_points[-1][0]:
                            min_points[-1] = [window_min, global_min_idx]
                            if show_diagnostics:
                                print('Replaced points')
                    else:         
                        min_points.append([window_min, global_min_idx])
                        if show_diagnostics:
                            print(f"{global_min_idx} added to list")
                else:
                    if show_diagnostics:
                        print("No point added")


        else:
            if show_diagnostics:
                print('Index on the edge of frame')

        start_index = start_index + int(window_shift*window_size)

    min_points = np.asarray(min_points)
    sorted_idx = np.argsort(min_points[:,0])
    min_points = np.asarray(min_points[sorted_idx])
    relative_val_diff = np.asarray((min_points[:,0] - min_points[0,0])/min_points[0,0])
    index_diff = np.asarray(abs(min_points[0,1]-min_points[:,1]))

    nearest_points = np.column_stack((min_points, relative_val_diff, index_diff))

    #---- Determine Variation ----#
    weighted_sum = 0
    weighting_diff = 0
    for i in range(nearest_points.shape[0]-1):
        w = 1-(weighting_diff*i)
        val_diff = nearest_points[i+1, 2]
        index_diff = nearest_points[i+1, 3]

        term = w * (alpha*index_diff + beta*(1/val_diff))
        weighted_sum += term

        if show_diagnostics:
            print(f"Val Diff: {val_diff} \t Index Diff: {index_diff}")
            print(f'Index Gap: {index_diff} \t Relative Diff From Min: {val_diff} \t Weighting: {w} \nTerm: {term} \t Index Contribution: {alpha*index_diff} \t Value Contribution: {beta*(1/val_diff)} \n')

    weighted_mean = weighted_sum/(nearest_points.shape[0]-1)

    return min_index, weighted_mean, nearest_points, y