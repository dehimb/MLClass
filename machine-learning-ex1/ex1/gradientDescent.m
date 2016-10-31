function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
% prepare vectorized data 
XV = [X,y];
theta = [theta; -1];
for iter = 1:num_iters

    tempTheta1 = theta(1) - alpha * 1/m * sum(XV * theta);
    tempTheta2 = theta(2) - alpha * 1/m * sum(XV * theta .* X(:,2));
    theta(1) = tempTheta1;
    theta(2) = tempTheta2;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta(1:2,:));

end
theta = theta(1:2,:);
end
