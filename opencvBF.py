import cv2
import numpy as np
from multiprocessing import Pool

def bilateral_filter (data, d, sigma_color, sigma_space, dtype=np.float32):
    filtered_data = np.zeros(data.shape, dtype)
    for i in range(data.shape[0]):
        filtered_data[i] = cv2.bilateralFilter(data[i], d, sigma_color, sigma_space)
    return filtered_data

def bilateral_filter_multiproc(data, d, sigmarange, sigmaspatial, processes, nchunks, dtype=np.float32):
    # Split the data into `nchunks` chunks along the 0th axis
    data_chunks = np.array_split(data, nchunks, axis=0)
    # Apply the bilateral filter to each chunk in parallel using a Pool of `processes`
    with Pool(processes) as pool:
        # Create a list of parameters for the `cv2.bilateralFilter` function to apply to each chunk
        parameters = [(chunk, d, sigmarange, sigmaspatial) for chunk in data_chunks]
        # Apply the bilateral filter to each chunk in parallel and store the results in `filtered_chunks`
        filtered_chunks = pool.starmap(bilateral_filter, parameters)
    
    # Combine the filtered chunks back into a single array and return it
    return np.concatenate(filtered_chunks, axis=0)