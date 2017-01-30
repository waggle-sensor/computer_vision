import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import time
import grad_hist
import pdb


def compute_grad(img):
    """
    :param img: w x h x 1 array
    :return: array r, array theta

    Takes in an w x h x 1 image (generally first layer in LUV color space) and return
    per-pixel gradient magnitude r and orientation theta
    """
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    r, theta = cv2.cartToPolar(grad_x, grad_y)
    return r, theta


def quantize_grad(r, t_bin, n_bins):
    """
    :param r: array of gradient magnitude
    :param t_bin: array of normalized orientations
    :param n_bins: int of gradient bins
    :return: arrays t_floor, t_ceil, r_floor, r_ceil

    Takes in gradient magnitude and quantized thetas. Outputs quantized
     r and theta. Returns both floor and ceiling values for interpolation
    """
    t_floor = np.floor(t_bin)
    t_ceil = np.fmod(t_floor + 1, n_bins)

    r_ceil = np.multiply((t_bin - t_floor), r)
    r_floor = r - r_ceil

    t_floor = np.int32(t_floor)
    t_ceil = np.int32(t_ceil)
    r_floor = np.float32(r_floor)
    r_ceil = np.float32(r_ceil)
    return t_floor, t_ceil, r_floor, r_ceil


def hog(img):
    """
    :param img: h x w x 1 image
    :return: histograms of gradients h and magntiude of gradients r
    Takes in an w x h x 1 image (generally first layer in LUV color space) and return
    gradient magnitude r and and histogram of gradients h.
    """
    cell_x = np.int64(4)
    cell_y = np.int64(4)
    n_bins = np.int64(6)

    r, theta = compute_grad(img)
    t_bin = n_bins * theta / (2 * np.pi)
    tf, tc, rf, rc = quantize_grad(r, t_bin, n_bins)
    h = grad_hist.grad_hist(tf, tc, rf, rc, cell_x, cell_y, n_bins)

    return h, r


def compute_chans(img):
    """
    :param img: h x w x 3 image
    :return: feature vector, not yet flattened
    This is the main function to call. We take in an RGB image and return
    all features for the image in an w x h x 10 array. Entries 0-2 are color space,
    entry 3-8 are gradient histograms, and entry 9 is gradient magnitude. We generally
    compute this for an entire image and then use a sliding window on this, which is why
    we don't flatten at this point.
    """
    shrink = np.int32(4)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)

    luv = grad_hist.sum_pool_frame(np.int32(x), shrink, shrink)

    h, r = hog(x[:, :, 0])
    r = grad_hist.sum_pool_grad(r, shrink, shrink)
    r = r.reshape(r.shape[0], r.shape[1], 1)
    feat_vec = np.concatenate((luv, h, r), axis=2)
    return feat_vec


def gen_gradient():
    """
    :return: pure gradent image
    This is a test function that outputs a pure 1-d gradient in a given
    orientation.
    """
    g = np.float32(0*np.ones((240, 240)))
    delta = 255
    for i in np.arange(0, 240, 1):
        g[i:i+1, :] += delta
        delta += -1

    delta = 0
    for i in np.arange(0, 240, 1):
        g[:, i:i+1] += delta
        delta += 0
    return g


def main():
    img = cv2.imread('images/test.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    # img = cv2.resize(img, (640, 480))
    # img = gen_gradient()
    # (h, w) = img.shape[:2]
    # center = (w / 2, h / 2)
    cv2.imshow('image', img/255)
    cv2.waitKey(2000)
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=np.nan)
    print("Image Shape: ", img.shape)
    start = time.time()
    for i in range(200):
        # print(i)
        h, r = hog(img[:, :, 0])
        # print(r.shape)
        r_s = grad_hist.sum_pool_grad(r, np.int32(4), np.int32(4))
        feat_vec = compute_chans(img)
    print("HOG Performance: ", 200 / (time.time() - start), "frames/s")
    print("Feature Vector Shape: ", feat_vec.shape)

    # Display the sum pooled histogram of gradients magnitudes
    #cv2.imshow('frame1', r_s/(16*255))
    cv2.imshow('frame1', feat_vec[:,:,9]/(16*255))

    # Display the scaled histogram of gradients magnitudes
    # cv2.imshow('frame2', cv2.cvtColor(img,cv2.COLOR_LUV2BGR))
    cv2.imshow('frame2', r/255)
    cv2.waitKey(0)

    # hog(img)
    # h = hog(img[:, :, 0])
    # print(h)


if __name__ == '__main__':
    main()
