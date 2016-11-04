function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


J = (1 / (2 * m)) * sum((X * theta - y).^2) + (lambda / (2 * m)) * sum((theta(2:end) .^2));

% calculate gradint and add refularization(use 0 value for theta0)
grad = (sum(repmat(X * theta - y, 1, size(theta,1)) .* X) / m)' + (lambda / m) .* [0; theta(2:end)];

% =========================================================================

end
