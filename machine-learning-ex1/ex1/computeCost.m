function J = computeCost(X, y, theta)
% Initialize some useful values
m = length(y); % number of training examples

% Prepare vectorize data
XV = [X,y];
thetaV = [theta; -1];

% compute cost function
J = 1/(2 * m) * sum((XV * thetaV).^2);

end
