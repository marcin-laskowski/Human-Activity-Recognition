%% CLASSIFICATION coursework 2

% %%%%%%%%%%%%%%%%%%%%%%%%%%%% K N N %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% to run the KNN main file following files are needed:
% Activites.mat
% ClassifyX.m (file from the ZIP)
% TrainClassifierX.m (file from the ZIP)
% SanityCheck.m

clear all
close all
clc


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load 'Activities.mat';

% number of classes
n = 2;

%% PREPARING THE DATA
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Chosing two classes
all_train_data = [train_data, train_labels];
j = 1;
for i = 1:length(all_train_data)
    if all_train_data(i,4) == 1 || all_train_data(i,4) == 2
        NEW_train_data(j,:) = all_train_data(i,:);
        j = j+1;
    else
    end
end
train_data = NEW_train_data(:,1:3);
train_labels = NEW_train_data(:,4);

all_test_data = [test_data, test_labels];
j = 1;
for i = 1:length(all_test_data)
    if all_test_data(i,4) == 1 || all_test_data(i,4) == 2
        NEW_test_data(j,:) = all_test_data(i,:);
        j = j+1;
    else
    end
end
test_data = NEW_test_data(:,1:3);
test_labels = NEW_test_data(:,4);


%% TRAINING AND TESTING
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% calculating parameteres
parameters = TrainClassifierX(train_data, train_labels);

% classifing test_data
best_predictions = ClassifyX(test_data, parameters);

% SanityCheck test
SanityCheck()

%% CALCULATING ACCURACY
% calculating accuracy of the model
compareMatrix = [test_labels, best_predictions];
for i = 1:length(compareMatrix)
    if compareMatrix(i,1) == compareMatrix(i,2)
        aMatrix(i,1) = 1; 
    else
        aMatrix(i,1) = 0;
    end
end

Accuracy = sum(aMatrix) / length(best_predictions)


