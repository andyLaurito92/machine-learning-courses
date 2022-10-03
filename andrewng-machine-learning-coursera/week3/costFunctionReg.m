function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


hypothesis = sigmoid(X * theta);
constant = 1 / m;
regularization_term = (lambda / (2 * m)) * sum((theta.^2)(2:size(theta,1)));
first_term = - y' * log(hypothesis);
second_term = (1 - y)' * log(1 - hypothesis);
J = constant * sum(first_term - second_term) + regularization_term;

grad_without_reg = constant * ((hypothesis - y)' * X);
grad = grad_without_reg' + (lambda / m) * theta;
grad(1) = grad_without_reg(1);
% =============================================================

end
