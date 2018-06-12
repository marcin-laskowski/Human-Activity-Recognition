%% CLASSIFICATION coursework 2
% %%% P R O B A B I L I S T I C  M O D E L  C L A S S I F I C A T I O N %%%
% to run the Naive bayes main file following files are needed:
% Activites.mat


clear all
close all
clc

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               P R E P A R A T I O N  O F  T H E  D A T A
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% loading training and testing data
load 'Activities.mat';

% number of classes
n = 2;

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


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clustering the data

all_train_data = [train_data, train_labels];
trainClusterData = {};

% Creating clusters for training
for k = 1:n
    j = 1;
    for i = 1:length(all_train_data)
        if all_train_data(i,4) == k
            trainClusterData{k}(j,:) = all_train_data(i,:);
            j = j+1;
        else
        end
    end
end


all_test_data = [test_data, test_labels];
testClusterData = {};

% Creating clusters for testing
for k = 1:n
    j = 1;
    for i = 1:length(all_test_data)
        if all_test_data(i,4) == k
            testClusterData{k}(j,:) = all_test_data(i,:);
            j = j+1;
        else
        end
    end
end


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            T R A I N I N G
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Probabilistic model classification 

% X - chosen point
% Ck - class [1, 2, 3 or 4]

% Naive Bayes classification
% P_Ck_X = P_X_Ck * P_Ck / P_X


% Porbability of the cluster (all points in the cluster / all training data)
P_C = {};
for i = 1:n
    P_C{i} = length(trainClusterData{i}) / length(train_data);
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% P(X|Ck)
% calculating mean of the each cluster
mu = {};
for i = 1:n
    mu{i} = [mean(trainClusterData{i}(:,1)), mean(trainClusterData{i}(:,2)), mean(trainClusterData{i}(:,3))];
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   C L A S S I F I C A T I O N
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% calculating sigma
sig = {};
for j = 1:n
    a12 = cov(trainClusterData{j}(:,1), trainClusterData{j}(:,2));
    a13 = cov(trainClusterData{j}(:,1), trainClusterData{j}(:,3));
    a21 = cov(trainClusterData{j}(:,2), trainClusterData{j}(:,1));
    a23 = cov(trainClusterData{j}(:,2), trainClusterData{j}(:,3));
    a31 = cov(trainClusterData{j}(:,3), trainClusterData{j}(:,1));
    a32 = cov(trainClusterData{j}(:,3), trainClusterData{j}(:,2));
    sig{j} = [var(trainClusterData{j}(:,1)), a12(1,2), a13(1,2); a21(1,2), var(trainClusterData{j}(:,2)), a23(1,2); a31(1,2), a32(1,2), var(trainClusterData{j}(:,3))];    
    a{j} = 1 / sqrt(det(2*pi * sig{j}));
end

%% Calculating Naive Bayes
P_X_Ck = {};
P_Ck_X = zeros(length(test_data),2);

for i = 1:length(test_data)
    X = test_data(i,1:3);
    for j = 1:n
        P_X_Ck{i,j} = a{j}*(exp(-1/2*(X-mu{j}) * inv(sig{j}) * (X-mu{j})'));
    end
    P_X = P_X_Ck{i,1} * P_C{1} + P_X_Ck{i,2} * P_C{2};
    
    P_Ck_X(i,1) = P_X_Ck{i,1} * P_C{1} / P_X;   
    P_Ck_X(i,2) = P_X_Ck{i,2} * P_C{2} / P_X;
end

%% Choosing best prediction

my_best_prediction = zeros(length(testClusterData),1);
for i = 1:length(test_data)
    a1 = P_Ck_X(i,1);
    a2 = P_Ck_X(i,2);
    a3 = 0;
    a4 = 0;
    
    if a1 > a2
        if a1 > a3
            if a1 > a4
                prediction = 1;
            else
                prediction = 4;
            end    
        else
            if a3 > a4
                prediction = 3;
            else
                prediction = 4;
            end 
        end
    else    
        if a2 > a3
            if a2 > a4
                prediction = 2;
            else
                prediction = 4;
            end 
        else
            if a3 > a4
                prediction = 3;
            else
                prediction = 4;
            end 
        end    
    end
    
    my_best_prediction(i) = prediction;
    
end




%% COMPARING TEST_LABELS with MY_BEST_PREDICTION

compareMatrix = [test_labels, my_best_prediction];

for i = 1:length(compareMatrix)
    if compareMatrix(i,1) == compareMatrix(i,2)
        aMatrix(i,1) = 1; 
    else
        aMatrix(i,1) = 0;
    end
end

Accuracy = sum(aMatrix) / length(my_best_prediction)
