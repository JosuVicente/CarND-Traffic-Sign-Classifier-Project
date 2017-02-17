#**Traffic Sign Recognition** 

##Writeup Template

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

[image1]: ./write_up/hist.png "Histogram"
[image2]: ./write_up/classes.png "Traffic Sign Classes"
[image3]: ./write_up/preprocess.png "Traffic Signs Preprocessed"
[image4]: ./write_up/augment.png "Traffic Sign augmented"
[image5]: ./write_up/histaug.png "Histogram After Augmentation"
[image6]: ./write_up/new.png "New Images"
[image7]: ./write_up/newclassified.png "New Images Classified"
[image8]: ./write_up/newsoftmax.png "New Images with Softmax Probabilities"
[image9]: ./write_up/accuracy.png "Accuracy"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This is it. Code [available here](https://github.com/JosuVicente/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fourth code cell of the IPython notebook.  

Firstly I display an histogram to show how many examples of each of the 43 classes we have in our training data

![alt text][image1]

And then I display a sample image for each of the classes

![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the sixth code cell of the IPython notebook.

For preprocessing the images I do two things:

Firstly I transform all images from 3 color channels to 1 averaged. The average is weighted based on the preceived brightness. This will also help the classifier as is less computationally expensive.

Secondly I normalize the data to values between -1 and 1 with mean 0 to reduce the variance and make things easier for the classifier.

Here are some example of original images and same images after preprocessing:

![alt text][image3]

Finally I shuffle the training set

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

For this project we were provided with training, validation and testing data.
The code for loading this data into our variables is in first code cell of the IPython notebook.  

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630

After taking a look to the histogram of training set I can see that some classes are clearly missrepresented. 
Also by taking a look at the type of signs I noticed that some of them can be flipped or rotated and still be a valid class so I decided to increase the training set by applying some flipping and rotation.

The seventh code cell of the IPython notebook contains the code for augmenting the data set. 

There are different options here:
* Images that can flip horizontally like "Bumpy Road" or "General Caution"
* Images that can flip vertically like "Speed Limit (30km/h)" or "No entry"
* Images that can flip horizontally and vertically like "Priority road" or "No vehicles"
* Images that can rotate 180 degrees like "End of all speed and passing..."
* Images that when flipped horizontally or vertically transform into a new class like "Keep right" or "Keep left"

Here is an example of an original image and an augmented image:
![alt text][image4]

And the histogram of our training examples after augmentation
![alt text][image5]

The difference between the original data set size and the augmented data set is the following 
Examples on the training set before: 34799 
Examples on the training set after: 62457

Some clasess keep misrepresented but I found that the accuracy increased when applying this technique.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the ninth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Preprocess         		| 32x32x1 RGB image   							| 
| Layer 1 - Convolutional     	| 5x5 patch, 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 				|
| Layer 2 - Convolutional     	| 5x5 patch, 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16 				|
| Flatten         		| outputs 400   							| 
| Layer 3 - Fully Connected     	| outputs 120 	|
| RELU					|												|
| Dropout     	| 0.5 	|
| Layer 4 - Fully Connected     	| outputs 84 	|
| RELU					|												|
| Dropout     	| 0.5 	|
| Layer 5 - Output Layer     	| outputs 43 	|



####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the thirteenth cell of the ipython notebook. 

To train the model, I used an Adam Optimizer and the following parameters:
Learning Rate: 0.001

Batch Size: 128

Epochs: 250

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the fourteenth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.947
* validation set accuracy of 0.969
* test set accuracy of 0.999

The accuracy of the training set is higher than the validation and test sets because the model was trained using this data so it should do well.
The accuracy of the validation set is also higher than the accuracy of the test set because the validation set was used to calculate the accuracy when training the model.

See below a chart showing the evolution of the accuracy against the validation set for each EPOCH:
![alt text][image9] 

The architecture chosen it's based on LeNet-5 implementation and I decided to do that because the problem is similar only with more classes and the input image data had same dimensions (32x32).

I decided to add dropouts and found the accuracy of the model to increase. By doing that we prevent overfitting.
The dropout value I selected was 0.5 and it's done on layers 3 and 4 (the fully connected layers).

A description of the layers is shown above.

The final model's accuracy on the training, validation and test show provide evidence of the model working well because all accuracies are high and even though values differ that's expected and they are close together meaning there is no high bias or high variance.

 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five traffic signs that I found on the web:

![alt text][image6] 

The fifth image might be difficult to classify because is not centered and the classifier does indeed fail if the number of EPOCHS is low.
The other images I expected the model to do good and it did.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the seventeenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy Road      		| Bumpy Road   									| 
| Keep Right     			| Keep Right 										|
| Yield					| Yield											|
| Priority Road	      		| Priority Road					 				|
| Speed Limit (30km/h)			| End of no Passing      							|

![alt text][image7] 

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This is more than accuracy of the test set but obviously 5 images are not enough to get conclussions.
When running less number of EPOCHS the fifth image is misclassified resulting on an accuracy of 80%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.

![alt text][image8] 

Image 1 (Bumpy road)

Top 5 probabilites: [  1.00000000e+00   1.54103184e-13   4.27531466e-23   7.22614437e-25
   2.02705995e-27]
   
Top 5 indexes: [22 38 26 29 24]

Image 2 (Keep right)

Top 5 probabilites: [ 1.  0.  0.  0.  0.]

Top 5 indexes: [38  0  1  2  3]

Image 3 (Yield)

Top 5 probabilites: [ 1.  0.  0.  0.  0.]

Top 5 indexes: [13  0  1  2  3]

Image 4 (Priority road)

Top 5 probabilites: [  1.00000000e+00   6.98530760e-16   1.94420941e-16   1.23793261e-17
   1.91082030e-24]
   
Top 5 indexes: [12 13 39 15  1]

Image 5 (Speed limit (30km/h))

Top 5 probabilites: [ 0.73438287  0.23592307  0.01351789  0.01249629  0.00168125]

Top 5 indexes: [ 1 36 38  6 14]

For the first four images the model is clearly sure of what image is with probabilities equal to 1 or almost 1.
The fifth image is not so sure and although it classify the image correctly the probability is only 0.73. As I said before this image with less EPOCHS is normally misclassified.

