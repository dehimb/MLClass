function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% Prepare vectorize data
XV = [X,y];
thetaV = [theta; -1];

% compute cost function
J = 1/(2 * m) * sum((XV * thetaV).^2);


% =========================================================================

end
