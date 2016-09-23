import json
import cv2
import glob
import re


def main():
    """
    This function is mostly responsible for processing the raw data from the Caltech data set.
    """

    data_dir = '/media/nolsman/TRANSCEND/data' #This path is obvious to my SSD, will need to change the path.
    train_dir = data_dir + '/train/positive'
    # train_dir = data_dir + '/train/positive_unscaled'
    sep = '/'
    aspect_ratio = .4
    thresh = .65
    res = [640, 480]
    # Pad images, these parameters are to match roughly what it seems the papers do
    #This padding makes it so there is some margin around the bounding boxes
    pad = [.12, .14]
    image_dir = data_dir + '/images'
    i = 0

    annotations = json.load(open(data_dir + '/annotations.json')) #the annotations containing the bounding boxes
    for set_dir in sorted(glob.glob(image_dir + '/*')):
        setID = re.search('set(\d)+', set_dir).group(0)

        for vid_dir in sorted(glob.glob(set_dir + '/*')):
            vidID = re.search('V(\d)+', vid_dir).group(0)

            for fname in sorted(glob.glob(vid_dir + '/*')):

                ind = re.search('(?<=\_)(\d)+(?=\.)', fname).group(0)
                if ind in annotations[setID][vidID]['frames'] and int(ind) % 30 == 29:
                    frame = cv2.imread(fname)
                    data = annotations[setID][vidID]['frames'][ind]
                    for datum in data:
                        if isinstance(datum['posv'], list):
                            bbv = datum['posv']
                            bb = datum['pos']
                            #Only keep bounding boxes with height greater than 50 pixels
                            if bb[3] > 50 and datum['lbl'] == 'person' \
                                    and check_bounds(bb) and compare_bb(bb, bbv, thresh):
                                i += 1
                                print(i)
                                bb_r = resize_bb(bb, aspect_ratio, res)
                                bb_p = pad_bb(bb_r, pad, res)
                                cv2.imwrite(train_dir + sep + 'frame' + str(i) + '.png', crop(frame, bb_p))


#
def check_bounds(bb):
    """
    :param bb:
    :return:
    This is used to make sure we only keep bounding boxes that are not too close to the edge of the frame
    """
    corners = [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]
    bounds = [5, 5, 635, 475]

    if corners[0] >= bounds[0] and corners[1] >= bounds[1] \
            and corners[2] <= bounds[2] and corners[3] <= bounds[3]:
        return True
    else:
        return False


def resize_bb(bb, aspect_ratio, res):
    """
    :param bb:
    :param aspect_ratio:
    :param res:
    :return:
    Resize bounding box to make sure it fits the aspect ratio
    """
    x, y, w, h = bb
    w_new = h*aspect_ratio
    x_new = max(x + (w - w_new)/2, 0)
    shift = min(res[0] - (x_new + w_new), 0)

    x_new += shift
    w_new += shift

    return x_new, y, w_new, h


def pad_bb(bb, pad, res):
    """
    :param bb:
    :param pad:
    :param res:
    :return:
    Pad bounding box
    """
    x, y, w, h = bb
    w_new = w * (1 + pad[0])
    h_new = h * (1 + pad[1])

    x_new = max(x + (w - w_new)/2, 0)
    y_new = max(y + (h - h_new)/2, 0)

    shift_x = min(res[0] - (x_new + w_new), 0)
    shift_y = min(res[1] - (y_new + y_new), 0)

    x_new += shift_x
    w_new += shift_x
    y_new += shift_y
    h_new += shift_y
    return x_new, y_new, w_new, h_new


def compare_bb(bb1, bb2, thresh):
    """
    :param bb1:
    :param bb2:
    :param thresh:
    :return:
    Essentially checks to make sure that the labeled bounding box and the visible bounding box
    are sufficiently similar
    """
    r = (bb2[2] * bb2[3]) / (bb1[2] * bb1[3])
    if r >= thresh or bb2 == [0, 0, 0, 0]:

        return True
    else:
        return False


def crop(frame, bb):
    """
    :param frame:
    :param bb:
    :return:
    Crops out region in bounding box from frame, rescales to 64x128
    """
    x, y, w, h = [int(v) for v in bb]
    c = cv2.resize(frame[y:y+h, x:x+w], (64, 128), interpolation=cv2.INTER_CUBIC)
    return c

def crop_unscaled(frame, bb):
    """
    :param frame:
    :param bb:
    :return:
    Crops out region in bounding boxe from frame, no rescaling
    """
    x, y, w, h = [int(v) for v in bb]
    c = frame[y:y+h, x:x+w]
    return c


if __name__ == '__main__':
    main()

