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

![Features](/ReadMe Images/features.png)
**Figure 1:** a typical example of feature extraction. Image from Dollar et. al 2014, IPAM

While there are many classes of features that can be extracted, a few have emerged as being robust for object detection. Higher-level filtering can be done on these features, as can be seen in the Filtered Channel Features paper, but the core of our pipeline will be the following:

* LUV Color Space
* Gradient Magnitude
* Histogram of Gradients

The code corresponding to this section is primarily located in channel_features.py, with some particular tasks implemented in a cython library grad_hist.pyx. These are mostly simply tasks, like iterating over sections of an image, which are slow in python but can be very efficient in cython.

###2.1 LUV Color Space

