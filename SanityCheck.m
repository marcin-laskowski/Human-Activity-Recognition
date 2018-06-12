function [] = SanityCheck()

assert(fopen('TrainClassifierX.m') > 0,'Could not find TrainClassifierX.m function file')
assert(fopen('ClassifyX.m') > 0,'Could not find ClassifyX.m function file')

% Each datapoint is described by 3 distinct features and labelled with a
% single integer value.
train_data = rand(100,3);
train_labels = randi([0,1],100,1);

% TrainClassifierX should accept such input
% There are no requirements regardsing the format of the parameters
% variable.
parameters = TrainClassifierX(train_data, train_labels);

disp('TrainClassifierX has been implemented correctly.')

test_data = rand(100,3);
predicted_labels = -1*ones(100,1);

% Fuction ClassifyX should take 3 features of a single datapoint and return 
% the predicted class (a single integer) to which the particular point belongs.
for i = 1:100
   predicted_labels(i,1) = ClassifyX(test_data(i,:), parameters);
end

disp('ClassifyX has been implemented correctly.')
disp('Sanity check passed!')

end