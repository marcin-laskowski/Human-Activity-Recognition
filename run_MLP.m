clear all
close all

%% Load Data
%
% data - readings from the accelerometer. Each column corresponds to 
% respectively X, Y and Z axis.
%
% labels - ID of the activity 
% 1 - walking
% 2 - running
% 3 - walking upstairs
% 4 - walking downstairs
%
% For binary classification you should change the labels of your chosen
% activities so there are only values 1 and 2. For example if you chose to
% classify walking upstairs and downstair you should change the labels 3
% and 4 to respectively 1 and 2 for binary classification to work
% correctly. When in doubt ask GTA.

load Activities.mat


%% Configurations/Parameters

% Network's architecture.
% Each element of the vector is the number of neurons in each hidden layer.
% For example:
% [1] - 1 hidden layer with 1 neuron
% [2 3] - 2 hidden layers with 2 and 3 neurons respectively
% Default MLP architecture: [5]
nbrOfNeuronsInEachHiddenLayer = [5];

% Epoch - one forward pass and one backward pass of all the training
% examples.
% Maximum number of epochs.
% Default number of epochs: 500.
nbrOfEpochs_max = 500;

%% Question 1A - How does the learning rate influence the training process?
% This is classic implementation of the backpropagation-based learning
% algorithm. All the methods in the function MLP_1A are covered in the
% lecture.
% Hint: You may have noticted that the learning is relatviely slow. You can
% try to run the code with more epochs to see at which epoch do the learning
% curves plateau. DON'T spend too much time on such experiments, as the
% process can be very time consuming. In the end you should report the
% results for 500 epochs.

% Learning rate
% Default learning rate: 0.0001
learningRate = 0.0001;

% Should learning rate decrease with each epoch?
enable_decrease_learningRate = 0; %1 for enable decreasing, 0 for disable
learningRate_decreaseValue = 0.0000001; % decrease value
min_learningRate = 0.00005; % minimum learning rate

[accuracy_1a, best_prediction_1a] = MLP_1A(train_data, train_labels, test_data, test_labels, nbrOfNeuronsInEachHiddenLayer, learningRate, nbrOfEpochs_max, enable_decrease_learningRate, learningRate_decreaseValue, min_learningRate);

% accuracy_1a - [nbrOfEpochs_max x 1] vector of accuracies obtained for each
% of training epochs. So-called 'learning curve'.
% 
% best_prediction_1a - [number of datapoints x 1] vector of predicted classes
% for each datapoint for the epoch that yielded the best accuracy. This can
% be directly compared with target_labels containing the true labels.

%% Rest of the questions
% Note that this function does not take learning rate as an input, as the 
% resilient gradient descent backpropagation (RPROP) is used in the training
% process. The reason for using it is better overall perfomance and much 
% faster runtimes.
% 
% The original paper describing the method:
% Martin Riedmiller, 'Rprop -Description and Implementation Details', 
% Technical Report, January 1994

[accuracy, best_prediction] = MLP_REST(train_data, train_labels, test_data, test_labels, nbrOfNeuronsInEachHiddenLayer, nbrOfEpochs_max);
 
% accuracy - [nbrOfEpochs_max x 1] vector of accuracies obtained for each
% of training epochs. So-called 'learning curve'.
% 
% best_prediction - [number of datapoints x 1] vector of predicted classes
% for each datapoint for the epoch that yielded the best accuracy. This can
% be directly compared with target_labels containing the true labels.