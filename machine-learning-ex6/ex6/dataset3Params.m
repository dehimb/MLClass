function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% test values
values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% create matrix with C, sigma pairs and errors for each pair
valuesPairs = zeros(length(values) ^ 2, 3);
% fill matrix
rowNumber = 1;
for i=1:length(values)
    for j=1:length(values)
        valuesPairs(rowNumber,:) = [values(i), values(j), 0];
        rowNumber = rowNumber + 1;
    end
end

for i=1:size(valuesPairs,1)
    model= svmTrain(X, y, valuesPairs(i,1), @(x1, x2) gaussianKernel(x1, x2, valuesPairs(i,2)));
    predictions = svmPredict(model, Xval);
    valuesPairs(i,3) = mean(double(predictions ~= yval));
end

[~,lowestErrorIndex] = min(valuesPairs(:,3));
lowestErrorPair = valuesPairs(lowestErrorIndex,:);
C = lowestErrorPair(1);
sigma = lowestErrorPair(2);



% =========================================================================

end
