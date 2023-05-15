import numpy as np
from scipy import signal, interpolate

def bilateral_numpy_fast(image, sigmaspatial, sigmarange, samplespatial=None, samplerange=None):
    """
    copy of https://github.com/OzgurBagci/fastbilateral/tree/master
    
    :param image: np.array
    :param sigmaspatial: int
    :param sigmarange: int
    :param samplespatial: int || None
    :param samplerange: int || None
    :return: np.array

    Note that sigma values must be integers.

    The 'image' 'np.array' must be given gray-scale. It is suggested that to use OpenCV.
    """
    # get height and width of input image
    height = image.shape[0]
    width = image.shape[1]
    
    # if no spatial and range sampling rates are provided, use standard deviation values
    samplespatial = sigmaspatial if (samplespatial is None) else samplespatial
    samplerange = sigmarange if (samplerange is None) else samplerange
    
    # flatten the image for easier indexing
    flatimage = image.flatten()
    
    # calculate the minimum and maximum pixel values of the image
    edgemin = np.amin(flatimage)
    edgemax = np.amax(flatimage)
    edgedelta = edgemax - edgemin
    
    # calculate derived spatial and range standard deviations for the kernel
    derivedspatial = sigmaspatial / samplespatial
    derivedrange = sigmarange / samplerange
    
    # calculate the padding size for the X and Y dimensions
    xypadding = round(2 * derivedspatial + 1)
    zpadding = round(2 * derivedrange + 1)
    
    # calculate the dimensions of the sample grid
    samplewidth = int(round((width - 1) / samplespatial) + 1 + 2 * xypadding)
    sampleheight = int(round((height - 1) / samplespatial) + 1 + 2 * xypadding)
    sampledepth = int(round(edgedelta / samplerange) + 1 + 2 * zpadding)
    
    # create a 1D array to hold the flattened image data
    dataflat = np.zeros(sampleheight * samplewidth * sampledepth)
    
    # create the X, Y coordinates for the sample grid
    (ygrid, xgrid) = np.meshgrid(range(width), range(height))
    
    # calculate the X, Y  dimensions for each pixel in the input image
    dimx = np.around(xgrid / samplespatial) + xypadding
    dimy = np.around(ygrid / samplespatial) + xypadding
    dimz = np.around((image - edgemin) / samplerange) + zpadding
    
    # flatten the X, Y dimensions into a 1D array
    flatx = dimx.flatten()
    flaty = dimy.flatten()
    flatz = dimz.flatten()
    
    # calculate the index for each pixel in the 1D array
    dim = flatz + flaty * sampledepth + flatx * samplewidth * sampledepth
    dim = np.array(dim, dtype=int)
    
    # fill in the 1D array with the pixel values from the input image
    dataflat[dim] = flatimage
    
    # Reshape the flattened data into a 3D matrix
    data = dataflat.reshape(sampleheight, samplewidth, sampledepth)
    weights = np.array(data, dtype=bool)
    
    # Set the dimensions of the kernel
    kerneldim = derivedspatial * 2 + 1
    kerneldep = 2 * derivedrange * 2 + 1
    halfkerneldim = round(kerneldim / 2)
    halfkerneldep = round(kerneldep / 2)
    
    # Create a meshgrid of x, y, and z values for the kernel
    (gridx, gridy, gridz) = np.meshgrid(range(int(kerneldim)), range(int(kerneldim)), range(int(kerneldep)))
    # Center the meshgrid values around zero
    gridx -= int(halfkerneldim)
    gridy -= int(halfkerneldim)
    gridz -= int(halfkerneldep)
    
    # Calculate the distance from the center of the kernel for each point in the grid
    gridsqr = ((gridx * gridx + gridy * gridy) / (derivedspatial * derivedspatial)) \
        + ((gridz * gridz) / (derivedrange * derivedrange))
    # Create a 3D Gaussian kernel based on the distance values
    kernel = np.exp(-0.5 * gridsqr)
    
    # Convolve the data and the weights with the kernel to blur the data
    blurdata = signal.fftconvolve(data, kernel, mode='same')
    blurweights = signal.fftconvolve(weights, kernel, mode='same')
    
    # Replace zero-valued weights with -2 to avoid division by zero
    blurweights = np.where(blurweights == 0, -2, blurweights)
    # Normalize the blurred data using the blurred weights
    normalblurdata = blurdata / blurweights
    # Replace any negative weights with zero to avoid negative values in the output
    normalblurdata = np.where(blurweights < -1, 0, normalblurdata)
    
    # Create a meshgrid of x and y values for the output
    (ygrid, xgrid) = np.meshgrid(range(width), range(height))
    
    # Calculate the dimensions of the output
    dimx = (xgrid / samplespatial) + xypadding
    dimy = (ygrid / samplespatial) + xypadding
    dimz = (image - edgemin) / samplerange + zpadding

    return interpolate.interpn((range(normalblurdata.shape[0]), range(normalblurdata.shape[1]),
                               range(normalblurdata.shape[2])), normalblurdata, (dimx, dimy, dimz))