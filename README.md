# **Traffic Sign Recognition** 

A CNN-based Traffic Sign Classifier trained on 43 different traffic signs belonging to the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). The highlight of the project is the extensive data pre-processing and augmententation pipeline. The CNN is a vanilla [LeNet](http://yann.lecun.com/exdb/lenet/) architecture. The Classifier achieves 97.1% accuracy on the training dataset and 95.6% accuracy on the test dataset.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Basic summary of the data set yields the following:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32) 
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I first plotted a histogram of the training data set to identify the distribution of the different classes. The dataset was highly unbalanced with some classes having more than 2000 sample images and others having less than 250 images. 

![Histogram of training dataset][/figures/training_histogram.png]

For completeness, I also plotted the histograms of the class distributions of the validation and test sets (also unbalanced)

Validation            |  Test
:--------------------------:|:-------------------------:
![Histogram of Validation dataset](/figures/validation_histogram.png)    |     ![Histogram of Test dataset](/figures/test_histogram.png)

Unbalanced training datasets are [known](http://www.ele.uri.edu/faculty/he/research/ImbalancedLearning/ImbalancedLearning_lecturenotes.pdf) to cause problems in classification tasks. Several approaches to address training imbalance are proposed in the literature such as Sampling	methods, cost-sensitive methods and kernel-based learning frameworks. An excellent and friendly summary of Data Imbalance in training deep neural networks is available [here](https://www.jeremyjordan.me/imbalanced-data/). 

The approach to tackle this problem is described in the next step.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The pre-processing pipeline consisted of three steps:
* __Colorspace conversion from RGB to YCrCb and extracting the Y ('Luma') channel:__

[YCrCb](https://en.wikipedia.org/wiki/YCbCr) is one of multiple colour models that separate intensity from colour information, making it more lighting invariant than RGB. This is important for traffic signs that often have very variable lighting due to shadows, time of day etc. 

* __Contrast-Limited Adaptive Histogram Equalization (CLAHE) on Y channel:__

Histogram equalization is an image-processing technique that enhances the contrast of images. Adaptive Histogram Equalization (AHE) adapts to the pixel intensity distribution in the image, making it more effective for images with non-uniform pixel intensity distribution. However, AHE tends to overamplify the contrast in near-constant regions of the image, since the histogram in such regions is highly concentrated.  Contrast Limited Adaptive Histogram Equalization (CLAHE) mitigates this problem by limiting the contrast-amplification which limits the noise amplification. CLAHE is run on the Y-channel of all images in the training set. An example of what that looks like (in Grayscale) is shown below:

![CLAHE-processed images](/figures/CLAHE_images.png)

* __Data augmentation to address class imbalace:__

To address the class imbalance in the training dataset, I generated augmented images with the four random effects: Rotation, Shearing, Shifting and Zooming. The augmentation was applied to each image of the training set with the goal of making each class have atleast 2000 sample images.

Five sample images from the training set which have already undergone the previous two steps in the pre-processing pipeline:
![Before augmentation](/figures/pre-processed_5.png)

After augmentation on each image, 4 augmented images are generated. So a total of 5x5 = 25 images are generated from the original 5 images:

![After augmentation](/figures/Augmented_5.png)


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The final model architecture was a modified LeNet architecture as shown below:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 pre-processed image   							| 
| Convolution 32 5x5 filters     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding,  outputs 14x14x32 				|
| Convolution 64 5x5 filters     	| 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding,  outputs 5x5x64 				|
| Fully connected		|   1600 -> 240						|
| Fully connected		|   240 -> 168					|
| Fully connected		|   168 -> 43					|
| Output				| Softmax      									|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For training, I used a learning rate of 0.001, 30 Epochs and a Batch size of 128. I used the Adam optimizer for the training loss. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Due to the extensive training dataset pro-processing, even the relatively simple CNN model yielded good results:

* training set accuracy of 97.1%
* test set accuracy of 95.9%

#### Notes on model selection

I experimented with different CNN architectures, notably [Sermanet](https://www.researchgate.net/profile/Yann_Lecun/publication/224260345_Traffic_sign_recognition_with_multi-scale_Convolutional_Networks/links/0912f50f9e763201ab000000/Traffic-sign-recognition-with-multi-scale-Convolutional-Networks.pdf), different variants of [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) and [Inception](https://arxiv.org/abs/1409.4842). After much experimentation, I decided to go for a simple modified LeNet model and instead focus more efforts on the dataset pre-processing. 

The modified architecture notably does not use dropout. I investigated the training loss in Tensorboard in the earlier phase but found no overfitting. This could be due to the extensive dataset augmentation which leads to sufficient generalization. However, this aspect needs to be more carefully investigated.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Downloaded images](/figures/Downloaded images.png)

Two of the downloaded images do not belong to the training dataset (Traffic Calming and Shared Pedestrian & Bike Path) These two are  incorrectly detected (unsurprisingly).

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Shared Pedestrian and Bike Path     		| Dangerous curve to the right   									| 
| Priority road     			| Priority road										|
| Traffic calming area (verkehrsberuhigter bereich)					| Road work											|
| Pedestrians only      		| No vehicles					 				|
| General caution			| No passing    							|

The softmax scores of the model on the five images are:

![Predicted probabilities](/figures/predictions_histogram.png)

Strictly speaking, the model predicted 1 out of 5 images correctly. However, 2 of the images were not in the training set. 
_Further investigations will be done to understand the poor performance._

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

__Image: Shared Pedestrian & Bike path__

|    Probabilities     |     Class     |      Sign      |
| :------------------: |:-------------:| :-------------:|
|  0.998961    | 20 | Dangerous curve to the right | 
| 0.000493 | 25    | Road work | 
| 0.000192 | 40    |  Roundabout mandatory  |
| 0.000156 | 35    | Ahead only  |
| 0.000107 | 11    | Right-of-way at the next intersection | 
   
   
   
__Image: Priority Road__
   
|    Probabilities     |     Class     |      Sign      |
| :------------------: |:-------------:| :-------------:|
|  1.000000   | 12 | Priority road | 
| 0.000000 | 9    | No passing | 
| 0.000000 | 40    |  Roundabout mandatory  |
| 0.000000 | 35    | Ahead only  |
| 0.000000 | 15    | No vehicles | 

__Image: Traffic Calming__

|    Probabilities     |     Class     |      Sign      |
| :----------------: |:-------------:| :-------------:|
|  0.999080   | 25 | Road work  | 
| 0.000518  | 5    | Speed limit (80km/h) | 
| 0.000221 | 36    |  Go straight or right  |
| 0.000083 | 38    | Keep right  |
| 0.000071 | 13    | Yield | 

__Image: Pedestrians only__

|    Probabilities     |     Class     |      Sign      |
| :----------------: |:-------------:| :-------------:|
|  0.607307   | 15 | No vehicles  | 
| 0.227924  | 34    |Turn left ahead | 
| 0.099947 | 39   |  Keep left  |
| 0.048865 | 13    | Yield  |
| 0.014065 | 33    | Turn right ahead | 

__Image: General caution__
   
|    Probabilities     |     Class     |      Sign      |
| :----------------: |:-------------:| :-------------:|
|  0.888047   | 9 | No passing | 
| 0.111953  | 12    | Priority road | 
| 0.000000 | 40   |  Roundabout mandatory  |
| 0.000000 | 35    | Ahead only  |
| 0.000000 | 23    | Slippery road | 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Not done (yet)
