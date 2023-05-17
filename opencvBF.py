import cv2
import numpy as np
from multiprocessing import Pool

def bilateral_filter_multiproc(data, d, sigmarange, sigmaspatial, processes, nchunks, dtype=np.float32):
    # Split the data into `nchunks` chunks along the 0th axis
    data_chunks = np.array_split(data, nchunks, axis=0)
    
    # Apply the bilateral filter to each chunk in parallel using a Pool of `processes`
    with Pool(processes) as pool:
        # Create a list of parameters for the `cv2.bilateralFilter` function to apply to each chunk
        parameters = [(chunk, d, sigmarange, sigmaspatial) for chunk in data_chunks]
        # Apply the bilateral filter to each chunk in parallel and store the results in `filtered_chunks`
        filtered_chunks = pool.starmap(cv2.bilateralFilter, parameters)
    
    # Combine the filtered chunks back into a single array and return it
    return np.concatenate(filtered_chunks, axis=0)