# computer_vision

##Introduction

This pipeline is predomianntly based on the work of two papers. The first paper "Fast Feature Pyramids for Object Detection" by Dollar et al. 2014, describes a method to efficiently compute features at a large number of different scales. 

"Filtered Channel Features for Pedestrian Detection" is a recent paper that set a benchmark on real-time pedestrian detection, based largely on methods from the feature pyramid paper. Below I will describe what I have developed so far for the object detection pipeline. A MATLAB implementation of these algorithms can be found [here](https://bitbucket.org/shanshanzhang/code_filteredchannelfeatures/src/d47c952daf86/Checkerboards_CVPR15_codebase/?at=default).

This code relies on a fairly mature MATLAB tool box that Piotr Dollar has been developing for several years. It is unfortunately only in MATLAB (with some C++), and the code can be somewhat difficult to read. This documentation will be broken into four main sections, that largely correspond to the differnet .py files I have written so far. The sections are

1. Dependencies
2. Feature Extraction
3. Training
4. Classification

[This page](http://mp7.watson.ibm.com/ICCV2015/ObjectDetectionICCV2015.html) has a lot of good resources to get a broad perspective on efficent object detection, especially [this talk](http://mp7.watson.ibm.com/ICCV2015/slides/2015_12_11_1400_iccv_efficient_detection_tutorial.pdf). And here is a good [very recent review paper](http://arxiv.org/pdf/1602.01237.pdf) that covers the work I based this pipeline on (from some of the same authors).

##1. Dependencies
###1.1 External Libraries 
This library was all written in Python 3. Because I have been implementing most of this code from scratch, there aren't many external dependencies:
* [numpy](http://www.numpy.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [cython](http://cython.org/)
* [opencv](http://opencv.org/)

The first three are all easy to install, open cv requires a bit more work as it is essentially a giant collection of C++ wrappers. The easiest guide to installing it can be found [here](http://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/), this is what I used to get it working.

###1.2 Training Data
Much of all the training and processing code relies on the training data from the [Caltech Pedestrian Detection Data Set](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/). Dealing with the data directly is somewhat difficult because of its file format, but I have extracted everything onto an external SSD. The scripts use that particular file path still, so if you use this data you will need to change those paths when necessary. If at any time you try to run a script and end up with an empty set of data, it is probably because the path was wrong somewhere. 

##2 Feature Extraction

The first step in any classification/detection pipeline is feature extraction, assuming that your data wasn't already structured such that the features were already naturally represented. 

![Features](/readme-images/features.png)
**Figure 1:** a typical example of feature extraction. Image from Dollar et. al 2014, IPAM

While there are many classes of features that can be extracted, a few have emerged as being robust for object detection. Higher-level filtering can be done on these features, as can be seen in the Filtered Channel Features paper, but
the core of our pipeline will be the following:

* LUV Color Space
* Gradient Magnitude
* Histogram of Gradients

The code corresponding to this section is primarily located in ```channel_features.py```, with some particular tasks implemented in a cython library ```grad_hist.pyx```. These are mostly simply tasks, like iterating over sections of an image, which are slow in python but can be very efficient in cython.

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

The function to compute HOG features is shown below, extracted from ```channel_features.py```:
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

The final step, computing the gradient histograms, is what is very slow in native Python. Because we have to iterate over each pixel in each 4x4 grid of an image, then add it to the corresponding histogram, it is essentially unavoidable to do all this in nested ```for``` loops. To this end, we perform all of these operations in Cython, as can be seen in the ```grad_hist.pyx``` file. You shouldn't ever have to edit anything in this file, but it may be worth looking at to get a sense of why some data types are structured the way they are. 

Because we pool on 4x4 grids with 6 bins, the output h as dimensions W/4 x H/4 x 6, where W and H are the width and height of the image, respectively.

###2.4 Feature Aggregation
It turns out that it is possible to substantially compress the feature space without losing much performance. For example, the six HOG features are already compress based on their 4x4 bins. We can also do something analogous with the three color features and one gradient magnitude feature by simply summing over 4x4 regions of the image, a technique known as sum pooling. In total this reduces the feature space by a factor of 4 x 4 = 16, without sacrificing performance.

With all this in place, we can simply call the function ```compute_chans()``` to take an image and extract its feature channels. Later parts of the pipeline essentially only use this function.


###2.5 Filtered Channles (future work)
So far we've discussed what we might think of as core features, which fit into a framework that Dollar *et al.* refer to as Aggregated Channel Features (ACF). While we can do classification directly on these features, one might imagine that further processing on the features may improve detection, in the same way that adding gradient features improves on the original color features. Zhang *et al.* showed that accuracy can be improved with Filtered Channel Features (FCF). Zhang tested a variety of filters, and found that a class they call Checkerboard filters yieled the greatest improvement.

![Frame](/readme-images/filters.png)

**Figure 4:**  Examples of filters tested by Zhang *et al.*

The downside is that more filters means a larger feature vector per frame, which means slower processing, training and prediction. Once the rest of the pipeline is functional, it will be worth evaluating the performance tradeoffs of adding these filters after the core feature extraction described above.


##3 Training

###3.1 Data Processing
Training is performed on the Caltech Pedestrian Detection Data Set mentioned in the introduction. The code to process the raw data is in ```process_data.py```, however if you are working from the directories I made you should not ever have to deal with this script because the relevent data is extracted elsewhere in the directory ```data\train\```. Because many adjacent frames are extremely similar, the data set is defined to consist of every 30th frame from all the videos.

![Frame](/readme-images/frame.png)

**Figure 5:** An example of a 640x480 frame from the Caltech data set.

The folder ```positive``` contains all the windows containing pedestrians, extracted from the raw images and resized to 64x128 for training purposes. ```positive_unscaled``` contains the same images, just not resized. The former is useful for training, the latter is useful for computing feature pyramids (described later). Note that we filter some of the positives, as it has been found that samples that are too small (say less than 50 pixels in height) become too distorted to be useful for training.

![Frame](/readme-images/pedestrian.png)

**Figure 6:** An example of a 64x128 pedestrian image from the Caltech data set.

For the first round of training, negatives are simply chosen at random from the training data. This is done by randomly selecting bounding boxes at different scales from the frames, and throwing out any that have too much overlap with the known positives. To limit redundancy, we take no more than 25 negatives per frame.

###3.2 Machine Learning/Adaboost

Now that we have extracted training data and know how to transform raw images into feature vectors, we can attempt to learn how to classify positive and negative examples. To do this we use the [Adaboost](https://en.wikipedia.org/wiki/AdaBoost) classifier. This is what is known as an ensemble model, essentially one made up of many weak classifiers, say depth-two [decision trees](https://en.wikipedia.org/wiki/Decision_tree_learning), where each individual classifer only does slightly better than random guessing. If dozens, hundreds, or even thousands of these weak classifiers are combined, they can perform much better than a single strong classifier. The strenght is in weighting the trees such that larger weights are assigned to the strongest classifiers, which can be done in a systematic way. For our purposes, all we care about is that the [scikit-learn](http://scikit-learn.org/stable/) library has this, and many other, models out of the box.

In the file ```train_model.py```, most of the functions just serve to load and format the training data. Below are a few exmaples of ```sklearn``` functionality, found in the ```train()``` function (bdt stands for boosted decision tree):

* Initialize model: ```bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=nweak)```
* Split data into 2/3 train and 1/3 validation sets: ```X1, X2, Y1, Y2 = train_test_split(X, y, test_size=0.33)```
* Fit model: ```bdt.fit(X1, Y1)```
* Make prediction of validation set: ```Yp = bdt.predict(X2)```
* Evaluate model accuracy: ```print(accuracy_score(Y2, Yp))```

Finally, we can save a model with

```joblib.dump(bdt, 'models/adaboost.pkl')```,

and load it again with

```bdt = joblib.load('models/adaboost.pkl')```

Training takes only a few seconds or so for a small number of classifiers, say ```n_estimators = 32```, however for larger models on the order of ```n_estimators = 256```, training can take upwards of 15 minutes. Dollar *et al.* and Zhang *et al.* ultimately used classifiers with thousands of trees, which can take many hours to train. They used a customized (and highly optimized) adaboost training procedure, so they report much faster training time than we get out-of-the-box with ```sklearn```. This will make testing difficult later on, however it will likely be feasible to still get good performance with far small models, as these papers were focusing on setting new benchmarks. 

###3.3 Hard Negative Mining/Bootstrapping

As in most binary classification problems, the previous sections relied on a set of labeled positive and negative samples on which to train. Our positives are generally well defined due to the labeling that comes with the data set, negatives are more ambiguous. Technically, any image that does not contain a pedestrian in it could be considered a negative sample. Imagine if our entire set of negatives consisted of largely monochrome images of the sky, one could imagine that our classifier may just learn that anything with mixed color is a pedestrian.
 
The solution to this problem is to pick negatives that, as much as possible, are representative of the scene in which the positive images (in this case pedestrians) would appear. In a perfect world, for each positive image containing a pedestrian we would be able to obtain the identical image of the exact same scene without the pedestrian. This is of course practically impossible, so we can use a technique called hard negative mining to improve accuracy at test time. 

The way this is works is to train an initial Adaboost classifer as described in section 3.2, with randomly selected negatives and a small number of weak classifiers, say ```n_estimators = 32```. This classifier may do a reasonable job of detecting pedestrians when they are in the frame (i.e. high precision), however it will also have a very high false positive rate. 

Here we will turn a bug into a feature, and use this detector to generate new negatives for the next round of training. This second round of training will use the same set of positives as the original model, however it will sample negatives both from the original negative set and the set of false positives found using the first model. These false positives are images that we know should be classified as negatives, yet managed to fool our original classifier, hence they are considered to be *hard negatives*. This second model will use a larger number of estimators as well, say ```n_estimators = 64```. 

This is a case of the more general statistical notion of [boostrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics), where we make use of the statistics of an existing data set to simulate the generation of more data. Ideally, as we repeat this process each new model will have asymptotically decreasing false positive rates. This comes at a price, since as each model improves it will become increasingly difficult to find false positives. 

Something to note is that, in earlier sections, the main performance bottlenecks were feature extraction and train time. Now, since we are using a trained model to feedback on the training process, prediction time also can constrain performance. In the next section we will see how to deal with this issue.


##4 Detection
Here we will get into some of the details of doing detection on whole images, and some methods that have been found to accelerate this process so that the overall framerate improves and that the bootstrapping described in section 3.3 is tractable. In terms of the actual code in this library, these two methods are still being developed and are works on progress. 

###4.1 Cascade Classification

As mentioned at the end of section 3, we noted that prediction time can be a crucial factor. Since the models may have many hundreds or even thousands of classifers, evaluating each one and aggregating their outputs can be extremely costly. To mitigate this, we can look at the other side of the coin of the idea of hard negatives - specifically that many negatives will actually be very easy to classify. 

If we have a classifier that has, for example, ```n_estimators = 256```, we don't necessarily need all 256 estimators to decide that a frame that only contains blue sky isn't a pedestrian. We can instead iteratively evaluate the estimators and develop some thresholds. For instance, we can say that if is scored as a negative with probability greater than .7 at any time, then evaluation can terminate early. Because adaboost estimators are evalauted in order of their predictive power, it is often the case that anything that is confidently decided to be a negative early will end up actually being a negative. This means that we may be able to classify most negatives with fewer than 10 estimators, and only use the full 256 for true positives or negatives that are exceedingly similar to a true positive (say a far off tree that strongly resembles the shape of a person). 

There are a few methods for doing this type of thresholding. One of which is called [Waldboost](http://cmp.felk.cvut.cz/~matas/papers/sochman-waldboost-cvpr05.pdf), which has fairly strongly theoretical grounding. More detail can be found in the author's [PhD thesis](http://cmp.felk.cvut.cz/ftp/articles/sochman/Sochman-TR-2009-24.pdf). The downside is that it is somewhat complex to implement, and Dollar states that it doesn't yield large improvement over a hand-tuned threshold. Unfortunately there isn't great documentation on how to do this sort of thresholding, the only resources appears to be some sparse documentation developed by Dollar and used by Zhang. Once the rest of the pipeline works, this threshold can be experimented with to see what yields acceptable results for a given application.

Conceptually this thresholding is quite simple, however it has proven to be tricky to implement efficently. ```sklearn``` is optimized to make many predictions at once, so ideally we will look at all of the windows for a given frame, extract their features, and make prediction at the same time. This is much faster than making individual predictions for each window. This is straightforward if we are using the standard form of prediction (i.e. ```bdt.predict(X)```), however this cascade prediction is (as far as I know) not built into ```sklearn```.

Because of this, we need to call the individual estimators and evalute their outputs. This becomes complicated when we want to perform this early termination, as we don't want to keep evaluating window that have already been classified as negatives. I tried implementing this in cython, however there wasn't much of a speedup because I had to call some much python code through ```sklearn```. The best method may be to stack all of the windows as we are already doing, and prune this list as negatives are identifeid by thresholding. The trick will be to make sure that the overhead from pruning doesn't undo all the gains from the cascade process. If pruning does prove to be a bottleneck it might make sense to do batch evaluation, say evaluate 8 estimators, prune, etc. rather than pruning after each classification.

An alternative is to implement adaboost from scratch and to build this functionality in from the beginning, however this is likely to be much worse than whatever has been optimized in the ```sklearn``` package.

###4.2 Feature Pyramids

Throughout the discussion so far, we've taken for granted the assumption of a fixed model size, i.e. 64x128 windows. Obviously it won't be the case that every pedestrian happens to fit precisely in this window, so it is necessary to do detection at multiple scales. This is where the idea of image pyramids come in, where we can use the same model size and simply resize the image to multiple scales. The naive implementation is simple, you simple interpolate the image up and down at say 24 scales, and then do object detection at each one. 

The issue is that this means that for each frame, you have now increased the resources required for object detection by a factor of 24. The paper from Dollar *et al.* proposes a novel technique where instead of computing image pyramids, we compute feature pyramids. More precisely, we first compute the features on the image and rescale those, which means that we don't need to recompute them at each scale. 

While this idea seems intuitively straight forward, the issue is that scaling an image by a factor of 1/2 does not imply that, for example, the gradient magnitude in the image scales by a factor of 1/2. Rather it has been shown that many common image features scale according to a power law. If the scales we are looking at are labeled ```s_1``` and ```s_2```, then the scaling of a given feature follow the relationship

```f(s_1)/f(s_2) = (s_1/s_2)^{-lambda},```

where ```lambda``` is compute from the statistics of example images. In practice this scaling only works over a limited range and works better for downsampling than for upsampling. Dollar use the strategy of interpolating over octaves, so explitly computing at original scale and say 1/2 scale, and interpolating with the power law relationship in between. This process can be seen in ```pyramids.py```. The computation of ```lambdas``` can be seen in the following function:

```
def compute_lambdas(frames, frames_s):
    n_chans = frames[0].shape[2]
    N = len(frames)

    f = np.array([[frames[i][:, :, c].mean() for c in range(n_chans)] for i in range(N)])
    f_s = np.array([[frames_s[i][:, :, c].mean() for c in range(n_chans)] for i in range(N)])

    mu_s = (f_s / f).mean(axis=0)
    lambdas = - np.log2(mu_s) / np.log2(1/2)
    return lambdas
```

This function takes in sets of frames and thsoe same frames rescaled by a factor of 1/2, and then estiamtes ```lambdas``` for each feature.
