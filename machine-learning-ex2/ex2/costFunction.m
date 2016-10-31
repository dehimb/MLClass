function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

z = X*theta;
templog(:,1) = log(sigmoid(z));
templog(:,2) = log(1-(sigmoid(z)));
tempy(:,1) = y;
tempy(:,2) = 1-y;
temp = templog.*tempy;

J = -(1/m)*(sum(temp(:,1))+sum(temp(:,2)));

grad(1:end,1)=(1/m)*(X' * (sigmoid(z)-y));

end
