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

---

### Data Set Summary & Exploration

#### 1. Data Set

I used the numpy library to calculate the following:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Visualize the dataset

To first thing to do in understanding the dataset is to see how many examples there are of each sign. Here is a bar graph showing the distribution of images in the training set (the validation and testing set have similar distributions):

![training set histogram]

I also looked at the distribution of colors for each sign. I averaged the RGB values for each pixel in each image type. I found that most signs had a dominate color. The speed limit signs had more red than anything else and others (like the "right turn only") had more blue. This helped me decide whether or not to grayscale the images.

### Design and Test a Model Architecture

#### 1. Pre-processing the data

I did not grayscale the image. I expiremented with grayscale but it didn't show any improvement over the full color images, so I left the images in full color.

I normalized the image data by subrtracing and dividing each pixel by 128. This causes the values to be between -1 and 1, which gives approximately zero mean. Normalizing the data helps the nueral network not have to deal with large biases.

Looking at the distribution of image types it was easy to see that there were a lot more examples of some signs than others. To make up for this, I generated extra images until each sign had an equal number of examples.

To generate the extra images, I took a random example of the sign and rotated it a random number of degress between -20 and 20.

Here is an example of an original image and an augmented image:

![original image]
![generated image]

The augmented data set contains 86430 images, more than twice the original amount.


#### 2. Model Architecture

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
 


#### 3. Training Parameters

To train the model, I used the Adam Optimizer to minimize the mean of the cross entropy loss. I used a batch size of 128. I originally had a learning rate of 0.0001, but I found a rate of 0.001 worked to get the same accuracy with fewer epochs. The network's accuracy stopped improving after around 7 epochs.

#### 4. Solution

My final model results were:
* training set accuracy of 0.991
* validation set accuracy of 0.955
* test set accuracy of 0.917

I began with the basic Lenet architecture. This architecture is known to work well with images. The convolution layers allow the network to pick up on finer details as well as large features of the image. This network did well with the initial training set, but was unable to get above 90% accuracy. Also, it shows signficant overfitting with the training accuracy about 10% higher than the validation accuracy.

To combat the overfitting, I added dropout to the network. Each convolution layer has a dropout rate of 20% and each fully connected layer has a dropout rate of 50%. This prevented the model from memorizing the training set, and helped with the overfitting, but hurt the overall accuracy a little bit.

The biggest help to getting a high enough accuracy was augmenting the dataset as described previously. Doing so increased the accuracy by almost 10% and decreased the gap between the training and validation accuracy. There is still some overfitting but the gap is now only about 5%.

Further improvements could be made to the network to improve accuracy, such as adding extra layers. But any additional layers would add more parameters and likely increase the overfitting problem. Likely, what would help the most is more refined techniques for data augmentation (adding in noise, shifting the image, etc).

 
### Test a Model on New Images

#### 1. Images found on the internet

Here are five German traffic signs that I found on the web:

<img src="https://github.com/dgard8/Project_2/blob/master/downloaded-signs/70kmh.png" width="120">
<img src="https://github.com/dgard8/Project_2/blob/master/downloaded-signs/120km.jpg" width="120">
<img src="https://github.com/dgard8/Project_2/blob/master/downloaded-signs/keepRight.jpg" width="120">
<img src="https://github.com/dgard8/Project_2/blob/master/downloaded-signs/stop.png" width="120">
<img src="https://github.com/dgard8/Project_2/blob/master/downloaded-signs/noPassingOverTons.png" width="120">
<img src="https://github.com/dgard8/Project_2/blob/master/downloaded-signs/roadNarrowsOnRight.jpg" width="120">
<img src="https://github.com/dgard8/Project_2/blob/master/downloaded-signs/roadWork.png" width="120">


I cropped out just the sign and converted them to 32x32 images. The last three might be hard for the network to classify because they aren't actually pictures of signs; the sign is the entire image. The network was trained on pictures where the sign doesn't take up the whole image.

#### 2. Internet image accuracy

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 70 km/h      			| No passing   									| 
| 120 km/h      		| Speed limit (100km/h) 						|
| Keep Right			| Keep Right									|
| Stop	      			| Speed limit (70km/h)							|
| No passing over 3.3 tons| No passing      							|
| Road Narrows on Right	| General caution					 			|
| Road Work	      		| No passing					 				|


The model was only able to correctly guess 1 of the 7 traffic signs, which gives an accuracy of 14%. This does not compare very well to the accuracy of the test data and suggests severe overfitting. Or that the images used for training were not representative of the images I found on the internet.

Interestingly, my network seems to prefer to suggest No passing. it gave that answer for three of my internet images, none of which were No passing signs.

During one of my earlier tests I did manage to get 5 of the 7 correct. But for some reason I can't reproduce that result.

#### 3. Probabilities for the internet images

For four of the images, the network is very unsure of the answer, with the highest probability at or below 10%. Here are the numbers for the 70 km/h sign:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .086         			| No passing   									| 
| .068     				| Speed limit (70km/h) 							|
| .059					| Speed limit (80km/h)							|
| .055	      			| Dangerous curve to the left					|
| .054				    | No passing for vehicles over 3.5 metric tons  |

##### Road narrows on right
For the "Road narrows on the right" sign, the network is very confident of its answer. But the answer is wrong, which is worse than being unsure. In all of my testing, this sign always had an almost 100% confidence in the wrong answer; it is unclear why. Here are the numbers:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| General caution   									| 
| .0006     			| Pedestrians 							|
| .0001					| Traffic signals							|
| .000002	      		| Right-of-way at the next intersection			|
| .00000008				 | Road narrows on the right  |


##### Keep right
For the "Keep right" sign, the network is very confident of its answer. This time the answer is correct. Here are the numbers:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Keep right   									| 
| .00003     				| Turn left ahead 							|
| .00000004					| Go straight or right							|
| .0000000001	      			| Roundabout mandatory					|
| .0000000001				    | Ahead only  |


##### 120 km/h
For the 120 km/h speed limit sign, the network is decently confident, but is wrong. It thinks the sign is for 100 km/h, and all the top five are for speed limit signs. Here are the numbers:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .76         			| Speed limit (100km/h)  									| 
| .20     				| Speed limit (120km/h) 							|
| .04					| Speed limit (80km/h)							|
| .002	      			| Speed limit (70km/h)					|
| .001				    | Speed limit (30km/h)  |