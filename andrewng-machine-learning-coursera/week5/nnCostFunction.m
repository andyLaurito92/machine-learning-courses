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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X_extended = [ones(m, 1) X];
z2 = Theta1 * X_extended';
a2 = sigmoid(z2);
a2_with_bias = [ones(size(a2, 2), 1) a2'];
a3 = sigmoid(a2_with_bias * Theta2');
[value, res] = max(a3, 0, 2);

constant = 1/m;
#yv = [1:num_labels] == y;
for k = 1:num_labels;
  y_k = y==k;
  h_theta_x = a3'(k, :);
  op1 = log(h_theta_x) * (-y_k);
  op2 = log(1 - h_theta_x) * (1 - y_k);
  J = J + constant * (op1 - op2);
endfor

regularization_constant = lambda / (2 * m);
theta1_without_bias = Theta1(:, 2:end);
theta2_without_bias = Theta2(:, 2:end);

regularized_term = regularization_constant * (sum(sum(theta1_without_bias .^2, 2)) + sum(sum(theta2_without_bias .^2, 2)));

J = J + regularized_term;


## Backpropagation algorithm
y_matrix = [1:num_labels] == y;
delta_3 = a3 - y_matrix;
delta_2 = (delta_3 * Theta2(:, 2:end)) .* sigmoidGradient(z2');

Delta2 = delta_3' * a2_with_bias;
Delta1 = delta_2' * X_extended;

Theta1_grad = Delta1 * constant;
Theta2_grad = Delta2 * constant;

## Adding regularization
constant_regularization_gradient = (lambda/m);
temp = Theta1_grad;
Theta1_grad = Theta1_grad + constant_regularization_gradient * Theta1;
Theta1_grad = [temp(:, 1) Theta1_grad(:, 2:end)];
	       
temp = Theta2_grad;
Theta2_grad = Theta2_grad + constant_regularization_gradient * Theta2;
Theta2_grad = [temp(:, 1) Theta2_grad(:, 2:end)];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
