function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

% prepare vectorized data
XV = [X,y];
tempTheta = [theta; -1];
theta = tempTheta;
for iter = 1:num_iters
    for i = 1:size(X,2)
        tempTheta(i) = theta(i) - alpha * 1/m * sum(XV * theta .* X(:,i));
    end
    theta = tempTheta;
    
    % ============================================================
    
    % Save the cost J in every iteration
    
    J_history(iter) = computeCost(X, y, theta(1:size(theta)-1,:));
    
end
theta = theta(1:size(theta)-1,:);
end
