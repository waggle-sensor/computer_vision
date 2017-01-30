import cv2
import numpy as np
import glob
import channel_features as cf


def load_images():
    """
    :return:
    Loads unscaled pedestrian frames, returns the feature channels for the frames at original and half scale.
    TODO: generalize to arbitrary scaling
    """
    images = []
    frames = []
    frames_s = []
    #pos_dir = '/media/nolsman/TRANSCEND/data/train/positive_unscaled'
    pos_dir = '/home/lane/Development/computer_vision/images'

    for fname in glob.glob(pos_dir + '/*'):
        img = cv2.imread(fname)
        channels = cf.compute_chans(img)
        channels_s = cf.compute_chans(cv2.pyrDown(img))
        frames.append(channels)
        frames_s.append(channels_s)
        images.append(img)

    return images, frames, frames_s


def compute_lambdas(frames, frames_s):
    """
    :param frames:
    :param frames_s:
    Takes in frames at original and half scale, computes lambdas for each feature.
    TODO: generalize to arbitrary scaling
    :return:
    """
    n_chans = frames[0].shape[2]
    N = len(frames)

    f = np.array([[frames[i][:, :, c].mean() for c in range(n_chans)] for i in range(N)])
    f_s = np.array([[frames_s[i][:, :, c].mean() for c in range(n_chans)] for i in range(N)])

    mu_s = (f_s / f).mean(axis=0)
    lambdas = - np.log2(mu_s) / np.log2(1/2)
    return lambdas


def scale_chans(frame, s, lambdas):
    """
    :param frame:
    :param s:
    :param lambdas:
    :return:
    Rescales channel features using computed values of lambda
    """
    assert s <= 1
    n_chans = frame.shape[2]
    print(lambdas)
    frame_s = np.array([cv2.resize(frame[:, :, c], None, fx=s, fy=s, interpolation=cv2.INTER_AREA) * s**(-lambdas[c])
                        for c in range(n_chans)])
    print(frame_s.shape)

    return frame_s


if __name__ == '__main__':
    images, frames, frames_s = load_images()
    print(len(frames))
    lambdas = compute_lambdas(frames, frames_s)
    #frame_index = 10
    frame_index = 7
    ft = frames[frame_index]
    print(frames_s[frame_index].shape)
    cv2.imshow('image', images[frame_index])
    test = scale_chans(ft, 1/2, lambdas)
    cv2.imshow('scaled', test[9, :, :]/(16*255))
    cv2.imshow('real', frames_s[frame_index][:, :, 9]/(16*255))
    cv2.waitKey(0)
    print(lambdas)
