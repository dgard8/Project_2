# **Traffic Sign Recognition** 

## Writeup

---
This document explains how I solved the problem of training a nueral network to identify German traffic signs. The code is [here](https://github.com/dgard8/Project_2/blob/master/Traffic_Sign_Classifier.ipynb).

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[training set histogram]: ./writeUpImages/trainDataHistogram.png "Traing set histogram"
[original image]: ./writeUpImages/originalImage.png "Original Image"
[generated image]: ./writeUpImages/generatedImage.png "Generated Image"

[70kmh]: ./downloaded-signs/70kmh.png "70 kmh"
[120kmh]: ./downloaded-signs/120kmh.jpg "120 kmh"
[keep right]: ./downloaded-signs/keepRight.jpg "keep right"
[no passing over tons]: ./downloaded-signs/noPassingOverTons.png "no passing over tons"
[road narrows on right]: ./downloaded-signs/roadNarrowsOnRight.jpg "road narrows on right"
[road work]: ./downloaded-signs/roadWork.png "road work"
[stop]: ./downloaded-signs/stop.png "stop"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I used the numpy library to calculate the following:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

To first thing to do in understanding the dataset it see how many examples there are of each sign. Here a bar graph showing the distribution of images in the training set (the validation and testing set have similar distributions):

![training set histogram]

I also looked at the distribution of colors for each sign. I averaged the RGB values for each pixel in each image type. I found that most signs had a dominate color. The speed limit signs had more red than anything else and others (like the "right turn only") had more blue. This helped me decide whether or not to grayscale the images.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I did not grayscale the image. I expiremented with grayscale but it didn't show any improvement over the full color images, so I left the images in full color.

I normalized the image data by subrtracing and dividing each pixel by 128. This causes the values to be between -1 and 1, which gives approximately zero mean. Normalizing the data helps the nueral network not have to deal with large biases.

Looking at the distribution of image types it was easy to see that there were a lot more examples of some signs than others. To make up for this, I generated extra images until each sign had an equal number of examples.

To generate the extra images, I took a random example of the sign and rotated it a random number of degress between -20 and 20.

Here is an example of an original image and an augmented image:

![original image]
![generated image]

The augmented data set contains 86430 images, more than twice the original amount.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the Lenet architecture for my network. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	|   1x1 stride, same padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 3x3     	|   1x1 stride, same padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 					|
| Fully connected		| output 120   									|
| Fully connected		| output 84   									|
| Fully connected		| output 43   									|
| Softmax				|         										|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam Optimizer to minimize the mean of the cross entropy loss. I used a batch size of 128. I originally had a learning rate of 0.0001, but I found a rate of 0.001 worked to get the same accuracy with fewer epochs. The network's accuracy stopped improving after around 7 epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

I began with the basic Lenet architecture. This architecture is known to work well with images. The convolution layers allow the network to pick up on finer details as well as large features of the image. This network did well with the initial training set, but was unable to get above 90% accuracy. Also, it shows signficant overfitting with the training accuracy about 10% higher than the validation accuracy.

To combat the overfitting, I added dropout to the network. Each convolution layer has a dropout rate of 20% and each fully connected layer has a dropout rate of 50%. This prevented the model from memorizing the training set, and helped with the overfitting, but hurt the overall accuracy a little bit.

The biggest help to getting a high enough accuracy was augmenting the dataset as described previously. Doing so increased the accuracy by almost 10% and decreased the gap between the training and validation accuracy. There is still some overfitting but the gap is now only about 5%.

Further improvements could be made to the network to improve accuracy, such as adding extra layers. But any additional layers would add more parameters and likely increase the overfitting problem. Likely, what would help the most is more refined techniques for data augmentation (adding in noise, shifting the image, etc).

 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

<img src="https://github.com/dgard8/Project_2/blob/master/downloaded-signs/70kmh.png" width="120">
<img src="https://github.com/dgard8/Project_2/blob/master/downloaded-signs/120km.jpg" width="120">
<img src="https://github.com/dgard8/Project_2/blob/master/downloaded-signs/keepRight.jpg" width="120">
<img src="https://github.com/dgard8/Project_2/blob/master/downloaded-signs/noPassingOverTons.png" width="120">
<img src="https://github.com/dgard8/Project_2/blob/master/downloaded-signs/roadNarrowsOnRight.jpg" width="120">
<img src="https://github.com/dgard8/Project_2/blob/master/downloaded-signs/roadWork.png" width="120">
<img src="https://github.com/dgard8/Project_2/blob/master/downloaded-signs/stop.png" width="120">


The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 


