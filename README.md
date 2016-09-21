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


###2.3 Oriented Gradients 
![Octagon](/readme-images/octagon-512.gif =256x256)
**Figure 2:** a typical example of feature extraction. Image from Dollar et. al 2014, IPAM
