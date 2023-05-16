import numpy as np
from scipy import signal, interpolate
import dxchange
import time


def bilateral_numpy_fast_3D(image, sigmaspatial, sigmarange, samplespatial=None, samplerange=None):
    """
    My adaptation of code https://github.com/OzgurBagci/fastbilateral/tree/master for 3D imagies
    
    :param image: 3D np.array representing a gray-scale image
    :param sigmaspatial: int representing the spatial standard deviation of the bilateral filter
    :param sigmarange: int representing the range standard deviation of the bilateral filter
    :param samplespatial: int representing the spatial sampling rate || None, defaults to sigmaspatial
    :param samplerange: int representing the range sampling rate || None, defaults to sigmarange
    :return: np.array representing the filtered image
    Note that sigma values must be integers.
    The 'image' 'np.array' must be given gray-scale. It is suggested that to use OpenCV.
    """
    start = time.time()
    # get height and width of input image
    xdim = image.shape[0]
    ydim = image.shape[1]
    zdim = image.shape[2]
    
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
    
    # calculate the padding size for the X and Y dimensions and the Z dimension
    xyzpadding = round(2 * derivedspatial + 1)
    rangepadding = round(2 * derivedrange + 1)
    
    # calculate the dimensions of the sample grid
    samplex = int(round((xdim - 1) / samplespatial) + 1 + 2 * xyzpadding)
    sampley= int(round((ydim - 1) / samplespatial) + 1 + 2 * xyzpadding)
    samplez= int(round((zdim - 1) / samplespatial) + 1 + 2 * xyzpadding)

    sampledepth = int(round(edgedelta / samplerange) + 1 + 2 * rangepadding)
    
    # create a 1D array to hold the flattened image data
    dataflat = np.zeros(samplex * sampley * samplez * sampledepth)

    # create the X, Y, Z coordinates for the sample grid
    (xgrid, ygrid, zgrid) = np.meshgrid(range(xdim), range(ydim), range(zdim))
    
    # calculate the X, Y, and Z dimensions for each pixel in the input image
    dimx = np.around(xgrid / samplespatial) + xyzpadding
    dimy = np.around(ygrid / samplespatial) + xyzpadding
    dimz = np.around(zgrid / samplespatial) + xyzpadding

    dimrange = np.around((image - edgemin) / samplerange) + rangepadding

    # flatten the X, Y, and Z dimensions into a 1D array
    flatx = dimx.flatten()
    flaty = dimy.flatten()
    flatz = dimz.flatten()   
    flatrange = dimrange.flatten()

    # calculate the index for each pixel in the 1D array    
    # A[x, y, z, w]=B[w + W*(z + D*(y + x*H))]

    dim = flatrange + sampledepth* (flatz + samplez * (flaty + flatx * sampley))
    dim = np.array(dim, dtype=int)

    # fill in the 1D array with the pixel values from the input image
    dataflat[dim] = flatimage

    # Reshape the flattened data into a 4D matrix
    data = dataflat.reshape(samplex, sampley, samplez, sampledepth)
    weights = np.array(data, dtype=bool)

    # Set the dimensions of the kernel
    kerneldim = derivedspatial * 2 + 1
    kerneldep = 2 * derivedrange * 2 + 1
    halfkerneldim = np.round(kerneldim / 2)
    halfkerneldep = np.round(kerneldep / 2)

    # Create a meshgrid of values for the kernel
    (gridx, gridy, gridz, gridrange) = np.meshgrid(np.arange(int(kerneldim)), np.arange(int(kerneldim)), np.arange(int(kerneldim)), np.arange(int(kerneldep)))

    # Center the meshgrid values around zero
    gridx -= int(halfkerneldim)
    gridy -= int(halfkerneldim)
    gridz -= int(halfkerneldim)
    gridrange -= int(halfkerneldep)

    # Calculate the distance from the center of the kernel for each point in the grid
    gridsqr = ((gridx * gridx + gridy * gridy + gridz * gridz) / (derivedspatial * derivedspatial)) + ((gridrange * gridrange) / (derivedrange * derivedrange))

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
    (xgrid, ygrid, zgrid) = np.meshgrid(np.arange(xdim), np.arange(ydim), np.arange(zdim), indexing='ij')

    # Calculate the dimensions of the output
    dimx = (xgrid / samplespatial) + xyzpadding
    dimy = (ygrid / samplespatial) + xyzpadding
    dimz = (zgrid / samplespatial) + xyzpadding

    dimrange = (image - edgemin) / samplerange + rangepadding
    return interpolate.interpn((range(normalblurdata.shape[0]), range(normalblurdata.shape[1]), range(normalblurdata.shape[2]), range(normalblurdata.shape[3])),
                              normalblurdata, (dimx, dimy, dimz, dimrange))


def find_min_max(data):
    """Find min and max values according to histogram"""
    h, e = np.histogram(data[:], 1000)
    stend = np.where(h > np.max(h)*0.0005)
    st = stend[0][0]
    end = stend[0][-1]
    mmin = e[st]
    mmax = e[end+1]
    return mmin, mmax
    
if __name__ == "__main__":
    f = dxchange.read_tiff_stack(f'data/recon_00000.tiff',ind=range(16))
    
    mmin,mmax=find_min_max(f[0])
    f[f<mmin]=mmin
    f[f>mmax]=mmax
    f = np.uint8(255*(f-mmin)/(mmax-mmin))
    fr = bilateral_numpy_fast_3D(f, 20, 40, samplespatial=None, samplerange=None)    
    dxchange.write_tiff_stack(fr.astype('float32'),f'data_re3D/recon.tiff',overwrite=True)