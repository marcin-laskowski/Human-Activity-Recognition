
function class = ClassifyX(input, parameters)

%% All implementation should be inside the function.

% For example: Random binary classification (returns 0 or 1 randomly).
% class = randi([0,1],1);

% define number of nearest points
K = 5;

% input, output data
test_data = input;
train_data = parameters(:,1:3);
train_labels = parameters(:,4);

% creating vector with large distances
for i = 1:K
    mindist(1,i) = 1000000;
end

% empty matrices
points = [];
best_predictions = [];
small_distance = mindist;
KNN = zeros(1,K);
ones = 0; 
twos = 0;
class = [];


for i = 1:length(test_data(:,1))
    small_distance = mindist;
    KNN = zeros(1,K);
    for j = 1:length(train_data)     
        % calculating the distance between test point and every 
        % point from the training data
%         points = [test_data(i,1), test_data(i,2), test_data(i,3); train_data(j,1), train_data(j,2), train_data(j,3)];
%         distance = pdist(points,'euclidean');
        point1 = test_data;
        point2 = train_data;
        distance = sqrt((point2(j,1) - point1(i,1))^2 + (point2(j,2) - point1(i,2))^2 + (point2(j,3) - point1(i,3))^2);
        
        % creating vector KNN eith dimensions (1,K) containing poistions
        % of the points with the smallest distances
        [max_dist, max_pos] = max(small_distance);
        if distance < max_dist
            small_distance(max_pos) = distance;
            KNN(max_pos) = j;
        end
    end

    % base on the position of the points - find a label which belongs to train
    % point
    for n = 1:K
        KNN_labels(n) = train_labels(KNN(n));
    end

    % calculating number of points which belongs to first label and second one.
    ones = 0;
    twos = 0;
    for m = 1:K
        if KNN_labels(m) == 1
            ones = ones + 1;
        else
            twos = twos + 1;
        end    
    end
 
    % comparing number of points which belongs to one and two label
    % and choosing the label for the test point
    if ones > twos
        best_predictions(i,1) = 1;
    elseif ones < twos
        best_predictions(i,1) = 2;
    else 
        % when there is the same number of labels from first cluster and from
        % the second cluster - sum the distance between the test point and the
        % all train points from one cluster and compare with the distance from 
        % the second cluster
        for incr = 1:K
            distance_1 = 0; distance_2 = 0;
            if KNN_labels(incr) == 1
                point = [test_data(i,1), test_data(i,2), test_data(i,3);...
                train_data(KNN(incr),1), train_data(KNN(incr),2), train_data(KNN(incr),3)];
                distance = pdist(point,'euclidean');
                distance_1 = distance_1 + distance;
            else
                point = [test_data(i,1), test_data(i,2), test_data(i,3);...
                train_data(KNN(incr),1), train_data(KNN(incr),2), train_data(KNN(incr),3)];
                distance = pdist(point,'euclidean');
                distance_2 = distance_2 + distance;
            end
        end
        % choose the smallest distance and assign label
        if distance_1 < distance_2
            best_predictions(i,1) = 1;
        else
            best_predictions(i,1) = 2;
        end  
    end    
end

class = best_predictions;

end


