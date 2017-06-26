**Traffic Sign Recognition** 

This is second project of Udacity Self Driving Car NanoDegree. 
A traffic sign classifier is trained over Germany traffic sign dataset using convolutional neural network.
---
[//]: # (Image References)

[image1]: ./30.jpg "30 speed limit"
[image2]: ./80.png "80 speed limit"
[image3]: ./no_entry.png "No entry"
[image4]: ./stop.jpg "Stop sign"
[image5]: ./road_work.jpg 
[image6]: ./Images_for_Readme/bar_chart.png "Bar chart for dataset visualization"
[image7]: ./Images_for_Readme/curve.png "Error curve"
[image8]: ./Images_for_Readme/example_image.jpg "Example training image"
[image9]: ./Images_for_Readme/visualize_featuremap.png "visualize featuremap"
### Load the data
Before running code, please run the download_data_set.sh shell script to download the dataset first!


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set by running download_data_set.sh file
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

More detailed visualization and summary of dataset can be viewed in [project code](https://github.com/fangyan93/Self_Driving_Car_Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier_1.ipynb).

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributes in terms of class label. The class label and corresponding meaning of the sign can be viewed in signnames.csv file.

![alt text][image6]
The x-axis refers to the class label, y-axis denotes the number of image in the training dataset.

Here is an example of a traffic sign image .

![alt text][image8]

In data preprocessing,
first, I normalize the data by dividing each pixel by 255, in this way, every element of training data lies in [0, 1], aiming at scaling the raw data and smooth the learningp process, otherwise, in each iteration, the weight will add a large term and the model may oscillate as a result.

Then, the data is centering by dividing each pixel by the sum of pixels in all training images, in this way, every element of training data lies in [-0.5, 0.5]. The reason for centering is similar to standardization, aiming at removing influcence of unbalanced data.

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

 


### Model summary

The RMSPropOptimizer optimizer is used in this project, with batch size of 125 and learing rate 0.001, over 50 epoches. 
The changes of training error and validation error is shown as the figure below, the red curve denotes trainiing accuracy, the black curve denotes the validation accuracy.
![alt text][image7]

### Training
All the weights in convolutional layer are initialized using xavier_initializer in tensorflow, this works for preventing weight from exploding or vanishing.
Regularization is added in loss function, in order to reduce overfitting. Without regularization, the model starts overfitting after 5-10 epoches.
My final model results were:
* training batch best accuracy of 1.0
* validation set accuracy of  0.9526
* test set accuracy of 0.9396

Convolutional neural network is famous for great performance on image classification and regression problems, therefore, a convoluional neural network is the first choice for this project.
A model with 2 convolutional layer and 1 fully connected layer was used at the begining, it was change to current one by adding 1 convolutional layer, also, the number of filter in each convolutional layer were increased, because the previous model or a smaller number of weights did not work well enough. In order to increase the accuracy, other than tuning hyperparameters, a good and essential way is to use more weight.
Learning rate is tuned at most in training, and it varied much for different optimizer. The RMSPropOptimizer optimizer works much better than GradientDescentOptimizer over this dataset, converging faster and at higher accuracy. Batch size is typical between 10-200, I choose 125, which is close to middle.

### Test a Model on New Images

Here are five German traffic signs that I found on the web:


![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

The first image might be difficult to classify because ...

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 speed limit      		| 30 speed limit  									| 
| 80 speed limit  			| Wild animals crossing										|
| Stop sign				| Stop sign											|
| Road Work       		| Road Work  				 				|
| No entry			| Wild animals crossing       							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. The performance of this model for this 5 images is worse than for the given dataset. The 

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

For the 5th image, the model is greatly sure that this is a sign of road work		(probability of 1.0), and the image does contains a road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.75         			| Wild animals crossing  									| 
| 0.1     				| Slippery road										|
| 0.15					| Speed limit (120km/h)										|
| 0.05	      			| No passing					 				|
| 0.05				    | General caution     							|

### Discussion:
We can see that for those 3 correctly classified images, the corresponding probability is 1, but for those 3 misclassified images, the largest probability are both about 0.75 and the label of top 5 probabilities are nearly all the same, which means the classifier is totally confused.
One highly probable reasion is that input image is not consistent with the dataset images of the all classes, in terms of image quality, lumination and contrast of the image, etc, i.e. these two images are not valid to some extent. The classifier has to make a choice without much confidence. That is why the probablity is not 1 but 0.75. But, anyway, it is still a little wired why both the misclassified images are recognized as the same class, and the probability for misclassified image should not as high as 0.75......

### Visualizing the Neural Network 
The visualization of feature maps for 1st and 2nd convolutional layers are shown in detail at the end of the [jupyter notebook](https://github.com/fangyan93/Self_Driving_Car_Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier_1.ipynb).

For the example sign of speed limit 30km/h image, we can see from output of 1st convolutional layer that the circle on that image and the central area of the sign are highlighted, the influence of background is reduce to some extent. From the output of 2nd convolutional layer, we still can vaguely see the circular pattern of the ouput. 

A screenshot of the visualization is shown as below
![alt text][image9]
