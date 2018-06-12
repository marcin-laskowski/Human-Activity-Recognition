# Human Activity Recognition
Machine Learning | Unsupervised Learning | MultiLayer Perceptron


### General Overview
The code is based on the Multilayer Perceptron approach to classify human activities dataset. It was implemented classifier to solve the problem at hand. Dataset is based on the human activity, which contains labeled recordings of four different activities collected from a smart phone accelerometer. The smart phone was worn in a trouser pocket. The accelerometer data (x,y,z acceleration) was collected at 20 Hz sample rate. The time-series were averaged into 3 dimensional data points that reflect the average accelerations over a 10 second-interval. Each data point has one of 4 possible classes: walking, jogging, walking upstairs, walking downstairs.

### Files
- *Activities.mat* file contains 4 variables. The original dataset has 10,000 data points of averaged linear acceleration values (X,Y, Z). This dataset is split into half for training data, contained in *train_data* and the other half for testing data, provided in *test_data*. These datasets was used to test the performance of both MLP and your own classification method. The columns of the variables *train_data* and *test_data* contain averaged linear acceleration values in the X,Y, Z axes respectively. The 4 recorded activities (classes) are walking, jogging, walking upstairs, walking downstairs with labels 1, 2, 3, 4 respectively. *train_labels* variable contains the labels for each data point in the *train_data* and *test_labels* variable contains the labels for each data point in the *test_data*.
- *run_MLP.m* – This is the script containing all the parameters of the multilayer perceptron and calls the *MLP.m* function that trains the network on the training data and returns the accuracy of the classification on the testing data.
-  *MLP.m* – The Multilayer Perceptron (MLP) function. This function takes in the parameters specified in *run_MLP.m* and trains the MLP on the training data and classifies the testing data in each epoch.
- *TrainClassifierX.m* – function which train classifier on the training data and return the parameters for classifier.
- *ClassifyX.m* – function script which take testing data with parameters and return the predicted class for each testing data point.
- *Classification_KNN_main* - KNN classification
- *Classification_Naive* - Naive bayes classification
- *SanityCheck.m* – This script checks if your two functions (ClassifyX , TrainClassifierX.m ) match the specifications we use. Note: If your functions do not pass the automatic tests performed by this function we will be unable to automatically evaluate their performance and may deduct marks

### Goal
The overall goal was to investigate how parameters influence the classification performance of the multilayer perceptron and compare the performance and efficiency of training with another classifier (General Rule: You should always compare your classifier to another classifier performing the same task).

### Results
- *Results.pdf* - File which contains investigation of the code in terms of different parameters
