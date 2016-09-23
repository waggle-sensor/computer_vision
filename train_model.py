import cv2
import json
import glob
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble.weight_boosting import _samme_proba
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import re
import channel_features as cf
import time
from slide_window import slide_window
import cascade as casc


def train(X, y, nweak=32):
    """
    :param X: training data
    :param y: training labels
    :param nweak: number of classifiers for adaboost
    :return:

    Trains adaboost classifier, this can take some time if nweak is large.
    """
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=nweak)

    X1, X2, Y1, Y2 = train_test_split(X, y, test_size=0.33)

    print('Fitting!')
    bdt.fit(X1, Y1)
    Yp = bdt.predict(X2)

    print('accuracy is:')
    print(accuracy_score(Y2, Yp))
    return bdt


def load_pos():
    """
    :return:
    Loads all positive training samples, automatically labeled 1
    """
    pos_dir = '/media/nolsman/TRANSCEND/data/train/positive'
    pos = []
    for fname in glob.glob(pos_dir + '/*'):
        img = cv2.imread(fname)
        x = cf.compute_chans(img).ravel()
        pos.append(x)
    X_p = np.array(pos)
    y_p = np.ones(len(pos))

    return X_p, y_p


def load_neg(N):
    """
    :param N:
    :return:
    Load random negatives, labels automatically set to 0
    """
    img_neg = random_negatives(N)
    neg = []
    for img in img_neg:
        x = cf.compute_chans(img).ravel()
        neg.append(x)
    X_n = np.array(neg)
    y_n = np.zeros(N)

    return X_n, y_n


def random_negatives(N):
    """
    :param N:
    :return:
    Mines  N random negatives from training data. Basically picks random bounding boxes
    and saves them if they don't overlap any positives too much.
    """
    data_dir = '/media/nolsman/TRANSCEND/data'
    aspect_ratio = .5
    nNeg = 25 #Max number of negatives per frame.
    lims = [0, 565, 0, 330, 25, 75]

    negatives = [None] * N
    annotations = json.load(open(data_dir + '/annotations.json'))
    image_locs = np.random.permutation(get_image_locs())
    fcount = 0
    ncount = 0
    while ncount < N:
        fname = image_locs[fcount]
        fcount += 1

        setID = re.search('set(\d)+', fname).group(0)
        vidID = re.search('V(\d)+', fname).group(0)
        ind = re.search('(?<=\_)(\d)+(?=\.)', fname).group(0)

        for j in range(nNeg):
            bb1 = random_bb(lims, aspect_ratio)
            neg = True
            frame = cv2.imread(fname)

            if ind in annotations[setID][vidID]['frames']:
                data = annotations[setID][vidID]['frames'][ind]
                for datum in data:
                    bb2 = datum['pos']
                    if iou(bb1, bb2) > .1: # Rejects if too much overlap
                        neg = False
            if neg:
                negatives[ncount] = crop(frame, bb1)
                ncount += 1
                if ncount % 25 == 0:
                    print(ncount)
                if ncount == N:
                    break
    return negatives


def hard_negatives(N, clf):
    """
    :param N: Number of negatives
    :param clf: classifier
    :return:
    Mines N hard negatives with the classifier clf. Basically searches for false positives.
    """
    data_dir = '/media/nolsman/TRANSCEND/data'

    negatives = [None] * N
    annotations = json.load(open(data_dir + '/annotations.json'))
    image_locs = np.random.permutation(get_image_locs())
    fcount = 0
    n_tot = 0
    while n_tot < N:
        fname = image_locs[fcount]
        fcount += 1

        setID = re.search('set(\d)+', fname).group(0)
        vidID = re.search('V(\d)+', fname).group(0)

        ind = re.search('(?<=\_)(\d)+(?=\.)', fname).group(0)
        img = cv2.imread(fname)
        frame = cf.compute_chans(img)
        wins, bbs = detect(frame, clf, 1)
        perm = np.random.permutation(np.arange(wins.shape[0]))
        wins = wins[perm, :]
        bbs = bbs[perm, :]
        ncount = 0

        for bb1, win in zip(bbs, wins):
            neg = True

            if ncount < 25:
                if ind in annotations[setID][vidID]['frames']:
                    data = annotations[setID][vidID]['frames'][ind]
                    for datum in data:
                        bb2 = datum['pos']
                        if iou(bb1, bb2) > .1:
                            neg = False
                if neg:
                    negatives[n_tot] = win
                    ncount += 1
                    n_tot += 1
                    if n_tot % 25 == 0:
                        print(n_tot)

                    if n_tot == N:
                        break

    return negatives


def random_bb(lims, aspect_ratio):
    """
    :param lims:
    :param aspect_ratio:
    :return:
    Generates a random bounding box with given aspect ratio within the ranges in lims.
    """
    x_min, x_max, y_min, y_max, w_min, w_max = lims
    x = np.random.randint(x_min, x_max)
    y = np.random.randint(y_min, y_max)
    w = np.random.randint(w_min, w_max)
    h = int(w / aspect_ratio)

    return x, y, w, h


def get_image_locs():
    """
    :return: image file names
    """
    locs = []
    data_dir = '/media/nolsman/TRANSCEND/data'

    image_dir = data_dir + '/images'
    for set_dir in sorted(glob.glob(image_dir + '/*')):
        for vid_dir in sorted(glob.glob(set_dir + '/*')):
            for fname in sorted(glob.glob(vid_dir + '/*')):
                ind = re.search('(?<=\_)(\d)+(?=\.)', fname).group(0)
                if int(ind) % 30 == 29:
                    locs.append(fname)

    return locs


def iou(bb1, bb2):
    """
    :param bb1:
    :param bb2:
    :return:
    Computes intersection-over-union metric for two bounding boxes
    """
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2
    left = max(x1, x2)
    right = min(x1 + w1, x2 + w2)
    top = max(y1, y2)
    bot = min(y1 + h1, y2 + h2)

    intersect = (right - left) * (top - bot)
    union = (w1 * h1) + (w2 * h2) - intersect

    return intersect / union


def crop(frame, bb):
    """
    :param frame:
    :param bb:
    :return:
    Crops a bounding box out of a frame, resizes to 64x128
    """
    x, y, w, h = [int(v) for v in bb]
    c = cv2.resize(frame[y:y+h, x:x+w], (64, 128), interpolation=cv2.INTER_CUBIC)
    return c


def detect(img, clf, c=1):
    """
    :param img:
    :param clf:
    :param c:
    :return:
    Performs sliding-window detection an a frame. Returns all bound boxes with class c
    """
    stride = 6
    w0, h0 = (16, 32)
    frames, bbs = slide_window(img, w0, h0, stride)
    # y = clf.predict(frames)
    y = cascade(frames, clf)
    bbs = np.int32(4 * bbs[y == c, :])
    frames = frames[y == c, :]
    return frames, bbs


def test(img, bdt):
    stride = 6
    w0, h0 = (16, 32)
    # img = cv2.pyrUp(img)
    # img = cv2.pyrUp(img)
    # img = cv2.pyrDown(img)
    clone = img.copy()
    # pyr = cv2.pyrUp(img)
    # start = time.time()
    img = cf.compute_chans(img)
    # print(time.time() - start)

    # start = time.time()

    frames, bbs = slide_window(img, w0, h0, stride)
    # print(time.time() - start)
    # f = frames[0,:100]
    # print(f)
    # start = time.time()
    y = bdt.predict(frames)
    cp = bdt.predict_proba(frames)
    print(cp.shape)
    print(.5 * np.log(cp[:, 0] / cp[:, 1]))
    # print(bdt.get_params())
    # print(time.time() - start)
    for yi, bb in zip(y, bbs):
        if yi == 1:
            x0, y0, x1, y1 = np.int32(4*bb)
            # print(bb)
            cv2.rectangle(clone, (x0, y0), (x1, y1), (0, 255, 0), 1)
    print(frames.shape)
    cv2.imshow('frame', clone)
    cv2.waitKey(0)
    # print(frames[-1,:,:,:])
    # print(bbs)
    # for p in range(3):
    #     h, w = pyr.shape[:2]
    #     clone = pyr.copy()
    #
    #     # print(w,h)
    #     for i in range(0, w - w0 - stride, stride):
    #         for j in range(0, h - h0 - stride, stride):
    #             win = pyr[j:j + h0, i:i + w0, :]
    #             # cv2.imshow('frame2',win)
    #             # print(i, i+w0)
    #             # print(j,j+h0)
    #             x = cf.compute_chans(win)
    #             y = bdt.predict(x.reshape(1, -1))
    #             # print(y)
    #             # cv2.imshow('frame', clone)
    #             if y == 1:
    #                 cv2.rectangle(clone, (i, j), (i + w0, j + h0), (0, 255, 0), 1)
    #                 # cv2.imshow('frame', clone)
    #                 # cv2.waitKey(50)
    #             # else:
    #             #     cv2.rectangle(clone, (i, j), (i + w0, j + h0), (0, 0, 255), 1)
    #             #     cv2.imshow('frame', clone)
    #             #     cv2.waitKey(50)
    #     cv2.imshow('frame', clone)
    #     cv2.waitKey(500)
    #     pyr = cv2.pyrUp(pyr)
        # pyr.append(cv2.pyrUp(pyr[-1]))


def cascade():
    """
    :return:
    Beginning of cascade classification. Still being developed.
    """
    pred = 0
    bdt = joblib.load('models/adaboost256.pkl')
    frame = cv2.imread('images/gradient.jpg')
    # X = cf.compute_chans(frame).ravel().reshape(1, -1)
    # X = cf.compute_chans(cv2.resize(frame, (64, 128))).ravel().reshape(1, -1)
    x = cf.compute_chans(cv2.resize(frame, (64, 128))).ravel()
    # np.random.shuffle(x)
    N = 100
    pred = [None] * N
    X = np.array([x for i in range(N)])
    start = time.time()
    for i in range(N):
        for t, estimator in enumerate(bdt.estimators_):
            pred += _samme_proba(estimator, 2, X)
            p_t = pred[0, 1] / (t + 1)
            print('p_t is: ', p_t)

            if p_t < -.2 and t > 8:
                return False

        return False
        out = casc.cascade(X, bdt)
    print((time.time() - start)/N)
    start = time.time()
    for i in range(N):
        out = bdt.predict(X)
    print((time.time() - start)/N)


    print(type(bdt))



def main():
    nN = 4000
    print('Getting Positives!')
    X_p, y_p = load_pos()
    print('Getting Negatives!')
    X_n, y_n = load_neg(nN)
    X = np.concatenate((X_p, X_n), axis=0)
    y = np.concatenate((y_p, y_n), axis=0)
    for i in range(5):
        bdt32 = train(X, y, 32)
        print('Mining Hard Negatives!')
        X_h = hard_negatives(nN, bdt32)
        X_h = np.concatenate((X_n, X_h), axis=0)
        samples = np.random.choice(np.arange(X_h.shape[0]), nN, replace=False)
        X_n = X_h[samples, :]
        print('Prediction on hard negatives:')
        y_h = bdt32.predict(X_n)
        print(accuracy_score(y_n, y_h))
        X = np.concatenate((X_p, X_n), axis=0)

    joblib.dump(bdt32, 'models/adaboostHN.pkl')


if __name__ == '__main__':
    # main()
    # train()
    cascade()
    # img = cv2.imread('test.png')
    # img = cv2.resize(img, (640, 480))
    # bdt32 = joblib.load('models/adaboost32.pkl')
    # bdt256 = joblib.load('models/adaboost256.pkl')
    # # start = time.time()
    # test(img, bdt32)
    # for i in range(100):
    # #     # print(i)
    #     test(img, bdt)
    # print((time.time() - start))
    # negs = random_negatives(5000)
    # for n in negs:
    #     cv2.imshow('negatives', n)
    #     cv2.waitKey(200)