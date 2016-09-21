# computer_vision

##Introduction

This pipeline is predomianntly based on the work of two papers. The first paper "Fast Feature Pyramids for Object Detection" by Dollar et al. 2014, describes a method to efficiently compute features at a large number of different scales. 

"Filtered Channel Features for Pedestrian Detection" is a recent paper that set a benchmark on real-time pedestrian detection, based largely on methods from the feature pyramid paper. Below I will describe what I have developed so far for the object detection pipeline. A MATLAB implementation of these algorithms can be found here:

https://bitbucket.org/shanshanzhang/code_filteredchannelfeatures/src/d47c952daf86/Checkerboards_CVPR15_codebase/?at=default

This code relies on a fairly mature MATLAB tool box that Piotr Dollar has been developing for several years. It is unfortunately only in MATLAB (with some C++), and the code can be somewhat difficult to read. This documentation will be broken into four main sections, that largely correspond to the differnet .py files I have written so far. The sections are

1. Dependencies
2. Feature Extraction
3. Training
4. Classification

This page has a lot of good resources to get a broad perspective on efficent object detection:

http://mp7.watson.ibm.com/ICCV2015/ObjectDetectionICCV2015.html

Especially this talk:

http://mp7.watson.ibm.com/ICCV2015/slides/2015_12_11_1400_iccv_efficient_detection_tutorial.pdf

And here is a good very recent review paper that covers the work I based this pipeline on (from some of the same authors):
http://arxiv.org/pdf/1602.01237.pdf

##1. Dependencies
###1.1 External Libraries 
This library was all written in Python 3. Because I have been implementing most of this code from scratch, there aren't many external dependencies:
* numpy
* sklearn
* cython
* open cv

The first three are all easy to install, open cv requires a bit more work as it is essentially a giant collection of C++ wrappers. The easiest guide to installing it can be found here, this is what I used to get it working:
http://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/

###1.2 Training Data
Much of all the training and processing code relies on the training data from the Caltech Pedestrian Detection Data Set:

http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/

Dealing with the data directly is somewhat difficult because of its file format, but I have extracted everything onto an external SSD. The scripts use that particular file path still, so if you use this data you will need to change those paths when necessary. If at any time you try to run a script and end up with an empty set of data, it is probably because the path was wrong somewhere. 

##2 Feature Extraction

The first step in any classification/detection pipeline is feature extraction, assuming that your data wasn't already structured such that the features were already naturally represented. 

![Features](/readme-images/features.png)
**Figure 1:** a typical example of feature extraction. Image from Dollar et. al 2014, IPAM

While there are many classes of features that can be extracted, a few have emerged as being robust for object detection. Higher-level filtering can be done on these features, as can be seen in the Filtered Channel Features paper, but
the core of our pipeline will be the following:

* LUV Color Space
* Gradient Magnitude
* Histogram of Gradients

The code corresponding to this section is primarily located in channel_features.py, with some particular tasks implemented in a cython library grad_hist.pyx. These are mostly simply tasks, like iterating over sections of an image, which are slow in python but can be very efficient in cython.

We compute 10 features per pixel over 64x128x3 images, with sum pooling (described later) that reduces the space by a factor of 16, resulting in a feature vector that has 64 * 128 * 10 / 16 = 5120 features per window. In reality we compute these features for the entire image, and then take 32x64x10 windows in feature space. In experiments on my laptop this runs at around 70 FPS for 640x480 images.

###2.1 LUV Color Space
LUV is an alternative to RGB color space, which apparently has been found to serve as better core features than RGB for object detection problems:

https://en.wikipedia.org/wiki/CIELUV

Like RGB, LUV consists of 3 features per pixel. Open CV has this conversion built in, so all we have to do is call

```img_LUV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LUV)```

For some reason Open CV reads in BGR instead of RGB, but so long as the image was loaded using Open CV you shouldn't have to worry about this detail. This color space seems to be used across a few papers, you could test others but should be safe not changing this aspect of the pipeline.

###2.2 Gradient Magnitude
There are many variants on the traditional RGB color space, e.g. HSV, LUV, etc. One of the earliest non-color space features that was focused on in the computer vision community is the image gradient. This allows us to look not at the absolute intensity of a given pixel, but how the intesnity changes from one pixel to the next. Gradients are particular useful because they are (ideally) independent of the background. On an 8-bit color scale, a transition from 5 to 10 will have the same gradient as a change of 205 to 210. Because many images have variable lighting, this invariance can be incredibly valuable. 

In section 2.3 we will discuss oriented gradients, but here we simply discuss the one-dimension feature of gradient magntiude. This can be computed quite simply and efficiently in Open CV using x- and y-oriented gradient computations. The entire function to compute this feature is 

```
def compute_grad(img):
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    r, theta = cv2.cartToPolar(grad_x, grad_y)
    return r, theta
```

```cv2.Sobel``` computes smoothed x and y gradients, which then are converted to polar coordinates. The ```r``` component of this representation is the pixel-wise gradient magnitude. We will use ```r``` and ```theta``` in the next section to compute oriented gradients.


###2.3 Histogram of Oriented Gradients (HOG)
While there is a great deal of useful information in the gradient magnitude, it doesn't tell the whole story. As you may have guess from the previous section, the orientation can also play a crucial role. Take Figure 2, if we tried to classify it purely based on gradient magnitude, we would lose all information about the geometry of the shape. If, however, we tracked both the gradient magnitude and direction, we would easily be able to detect the unique features corresponding to the eight edges in the image.

![Octagon](/readme-images/octagon-512.gif)

**Figure 2**

This is where the idea of a Histogram of Oriented Gradient (HOG) feature comes into play. The basic concept is to take a small window of the image, say 4x4 or 8x8, and bin the gradient magnitudes based on their orientation (see Figure 3). For example, if we used 4 bins, then we'd have one for 0, 90, 180, and 270 degrees. Within our 4x4 grid, we would add the gradient magnitude proportionally to its nearest bins. 

For example, if one of the 16 entries in the grad has ```r = 1``` and ```theta = 60```, then we would add .34 to the 0 degrees bin and .66 to the 90 degrees bin. Doing this for each of the entries in the grid, we get a histogram of orentied gradients feature. This is a very compact representation, as the original 4x4 grid had 4x4x2 = 32 floating point numbers, and the HOG has only N numbers (where N is the number of bins).


![HOG](/readme-images/hog.png)

**Figure 3:** Histogram of Oriented Gradients on 4x4 grids.

Following the papers mentioned in the introduction, we used 4x4 grids with N  = 6 histograms per histogram. Computing HOG features is substantially more complex than other features mentioned so far, as they cannot be computed from simple array operations in numpy and Open CV. The latter has some functionality to compute these features, however it is poorly documented.

The function to compute HOG features is shown below, extracted from channel_features.py:
```
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
```

This starts by using the ```compute_grad()``` function described in section 2.2 to get r and theta values for the entire image. The orientations are then normalized such that they range from 0 to n_bins, instead of from 0 to 2pi. These are then quantized in ```quantize_grad()```, which returns each theta rounded up and down along with a proportional distribution of the corresponding value of r. So far all of these operations can be efficiently computed by taking advantage of numpy's fast array operations. 

The final step, computing the gradient histograms, is what is very slow in native Python. Because we have to iterate over each pixel in each 4x4 grid of an image, then add it to the corresponding histogram, it is essentially unavoidable to do all this in nested ```for``` loops. To this end, we perform all of these operations in Cython, as can be seen in the grad_hist.pyx file. You shouldn't ever have to edit anything in this file, but it may be worth looking at to get a sense of why some data types are structured the way they are. 

Because we pool on 4x4 grids with 6 bins, the output h as dimensions W/4 x H/4 x 6, where W and H are the width and height of the image, respectively.

###2.4 Feature Aggregation
It turns out that it is possible to substantially compress the feature space without losing much performance. For example, the six HOG features are already compress based on their 4x4 bins. We can also do something analogous with the three color features and one gradient magnitude feature by simply summing over 4x4 regions of the image, a technique known as sum pooling. In total this reduces the feature space by a factor of 4 x 4 = 16, without sacrificing performance.

##3 Training

###3.1 Data Processing
Training is performed on the Caltech Pedestrian Detection Data Set mentioned in the introduction. The code to process the raw data is in process_data.py, however if you are working from the directories I made you should not ever have to deal with this script because the relevent data is extracted elsewhere in the directory ```data\train\```. 

The folder ```positive``` contains all the windows containing pedestrians, extracted from the raw images and resized to 64x128 for training purposes. ```positive_unscaled``` contains the same images, just not resized. These former is useful for training, the latter is useful for computing feature pyramids (described later). 

