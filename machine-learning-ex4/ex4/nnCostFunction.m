function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Add X0 to our features
X = [ones(m,1), X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Calculate feedforward for all training example to get a3
z2 = (Theta1 * X')';
a2 = sigmoid(z2);
% add bias unit to a2
a2 = [ones(m,1), a2];
z3 = (Theta2 * a2')';
a3 = sigmoid(z3);

% make cost calculations by parts
temp1 = log(a3);
temp2 = log(1-a3);
% prepare y for element wise multiplication
for i=1:size(y,1)
    for j=1:num_labels
        tempY(i,j) = j == y(i);
    end
end
costMatrix = temp1 .* tempY + (1-tempY) .* temp2;
% compute regularization term(exclude bias values from theta before operation)
trimmedTheta1 = (Theta1(:,2:end));
trimmedTheta2 = (Theta2(:,2:end));
r = (lambda / (2 * m))*(sum(trimmedTheta1(:).^2) + sum(trimmedTheta2(:).^2));
J = (-1/m) * sum(costMatrix(:)) + r;

% implemen backpropagation
for i=1:m
    % feedforward
    a1i = X(i,:)';
    z2i = Theta1 * a1i;
    a2i = [1;sigmoid(z2i)];
    z3i = Theta2 * a2i;
    a3i = sigmoid(z3i);
    % calculate errors
    yt = zeros(num_labels,1);
    yt(y(i)) = 1;
    d3i = a3i - yt;
    d2i = (Theta2' * d3i) .* sigmoidGradient([1;z2i]);
    Theta1_grad = Theta1_grad + d2i(2:end) * a1i';
    Theta2_grad = Theta2_grad + d3i * a2i';
end

% add regularization
Theta1_grad(:,1) = Theta1_grad(:,1) ./ m;
Theta1_grad(:, 2:end) = Theta1_grad(:,2:end) ./ m + (lambda / m) * Theta1(:, 2:end);
Theta2_grad(:,1) = Theta2_grad(:,1) ./ m;
Theta2_grad(:, 2:end) = Theta2_grad(:,2:end) ./ m + (lambda / m) * Theta2(:, 2:end);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
