**Traffic Sign Recognition** 

This is second project of Udacity Self Driving Car NanoDegree. A traffic sign classifier is trained over Germany traffic sign dataset using convolutional neural network.
Before running code, please run the download_data_set.sh shell script to download the dataset first!

---
[//]: # (Image References)

[image1]: ./30.jpg "30 speed limit"
[image2]: ./80.png "80 speed limit"
[image3]: ./no_entry.png "No entry"
[image4]: ./stop.jpg "Stop sign"
[image5]: ./road_work.jpg 
[image6]: ./Images_for_Readme/bar_chart.png "Bar chart for dataset visualization"
[image7]: ./Images_for_Readme/curve.png "Error curve"
[image8]: ./Images_for_Readme/example_image.png "Example training image"
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set by running download_data_set.sh file
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report



## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README
You're reading it! and here is a link to my [project code](https://github.com/fangyan93/Self_Driving_Car_Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier_1.ipynb)

###Data Set Summary & Exploration

####1. I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is [32, 32, 3]
* The number of unique classes/labels in the data set is 43
More detailed visualization and summary of dataset can be viewed in project code above.

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributes in terms of class label. The class label and corresponding meaning of the sign can be viewed in signnames.csv file.

![alt text][image6]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data.What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Here is an example of a traffic sign image .

![alt text][image7]

First, I normalize the data by dividing each pixel by 255, in this way, every element of training data lies in [0, 1], aiming at scaling the raw data and smooth the learningp process, otherwise, in each iteration, the weight will add a large term and the model may oscillate as a result.

Then, the data is centering by dividing each pixel by the sum of pixels in all training images, in this way, every element of training data lies in [-0.5, 0.5]. The reason for centering is similar to standardization, aiming at removing influcence of unbalanced data.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x32     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5x64	    | 1x1 stride, valid padding, outputs 10x10x64      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				|
| Convolution 3x3x128	    | 1x1 stride, valid padding, outputs 3x3x128      									|
| RELU					|												|
| Fully connected batch_size x 1152 | From output of 3rd convolution layer to output layer, output batch_size x 43      |
| Softmax		 	|apply softmax over output, i.e., the list of size 43 									|

 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The RMSPropOptimizer optimizer is used in this project, with batch size of 125 and learing rate 0.001, over 50 epoches. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

All the weights in convolutional layer are initialized using xavier_initializer in tensorflow, this works for preventing weight from exploding or vanishing.
Regularization is added in loss function, in order to reduce overfitting. Without regularization, the model starts overfitting after 5-10 epoches.
My final model results were:
* training batch best accuracy of 1.0
* validation set accuracy of  0.9526
* test set accuracy of 0.9396

Convolutional neural network is famous for great performance on image classification and regression problems, therefore, a convoluional neural network is the first choice for this project.
A model with 2 convolutional layer and 1 fully connected layer was used at the begining, it was change to current one by adding 1 convolutional layer, also, the number of filter in each convolutional layer were increased, because the previous model or a smaller number of weights did not work well enough. In order to increase the accuracy, other than tuning hyperparameters, a good and essential way is to use more weight.
Learning rate is tuned at most in training, and it varied much for different optimizer. The RMSPropOptimizer optimizer works much better than GradientDescentOptimizer over this dataset, converging faster and at higher accuracy. Batch size is typical between 10-200, I choose 125, which is close to middle.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:


![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 speed limit      		| 30 speed limit  									| 
| 80 speed limit  			| Wild animals crossing										|
| Stop sign				| Stop sign											|
| Road Work       		| Road Work  				 				|
| No entry			| Wild animals crossing       							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. The performance of this model for this 5 images is worse than for the given dataset. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in [project code](https://github.com/fangyan93/Self_Driving_Car_Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier_1.ipynb)

For the first image, the model is greatly sure that this is a sign of Speed limit (30km/h) (probability of 1.0), and the image does contain a Speed limit (30km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (30km/h)   									| 
| >.01     				| Speed limit (20km/h) 										|
| >.01					| Speed limit (70km/h)										|
| >.01	      			| Speed limit (50km/h)					 				|
| >.01				    | Speed limit (60km/h)      							|



For the second image, the model is relatively sure that this is a sign of Wild animals crossing		(probability of 0.75), and the image refers to a Speed limit (80km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.75         			| Wild animals crossing   									| 
| 0.1     				| Slippery road										|
| 0.075					| Speed limit (120km/h)										|
| 0.05	      			| No passing					 				|
| 0.05				    | General caution     							|


For the third image, the model is greatly sure that this is a stop sign		(probability of 1.0), and the image does contains a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop sign   									| 
| >.01     				| Bicycles crossing									|
| >.01					| No entry										|
| >.01	      			| Dangerous curve to the right					 				|
| >.01				    | End of speed limit (80km/h)      							|

For the 4th image, the model is greatly sure that this is a sign of road work		(probability of 1.0), and the image does contains a road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Road work  									| 
| >.01     				| No passing								|
| >.01					| Pedestrians									|
| >.01	      			| Speed limit (80km/h)					 				|
| >.01				    | Wild animals crossing     							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The visualization of feature maps for 1st and 2nd convolutional layers are shown in detail at the end of the [jupyter notebook](https://github.com/fangyan93/Self_Driving_Car_Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier_1.ipynb).

For the example sign of speed limit 30km/h image, we can see from output of 1st convolutional layer that the circle on that image and the central area of the sign are highlighted, the influence of background is reduce to some extent. From the output of 2nd convolutional layer, we still can vaguely see the circular pattern of the ouput. 

For the example image, which is as speed limit 30km/h  
For the example image, which is as speed limit 30km/h 
