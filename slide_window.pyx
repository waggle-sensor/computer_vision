import numpy as np
cimport numpy as np
import channel_features as cf
cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def slide_window(np.ndarray[DTYPE_t, ndim=3] img, int w_win, int h_win, int stride):
    """
    :param img:
    :param w_win:
    :param h_win:
    :param stride:
    :return:
    Takes in an image and performs a w_win x h_win sliding window on it. Returns each individual window
    in a flattened vector for classification, as well as the original bound box so that the frame can be
    located on the original image if necessary.
    """
    cdef:
        int i, j, x, y, c, n, i_c, j_c, ind
        int h = img.shape[0]
        int w = img.shape[1]
        int d = img.shape[2]
        int h_range = np.ceil((h - h_win)/stride)
        int w_range = np.ceil((w - w_win)/stride)
        np.ndarray[DTYPE_t, ndim=2] frames
        np.ndarray[DTYPE_t, ndim=2] bbs
    n = h_range * w_range

    frames = np.zeros((n, h_win * w_win * d), dtype=DTYPE)
    bbs = np.zeros((n, 4), dtype=DTYPE)
    ind = 0

    for j in range(h_range):
        for i in range(w_range):

            i_c = i*stride
            j_c = j*stride
            f = img[j_c:j_c +  h_win, i_c: i_c + w_win, :]
            frames[ind, :] = f.ravel() #flatten feature vector
            bbs[ind, :] = np.int32([i_c, j_c, i_c + w_win, j_c + h_win])
            ind += 1
    return frames, bbs
