function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
variableSize = size(z);  
g = zeros(variableSize);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
upTo = variableSize(1) * variableSize(2);
for i = 1:upTo;
  g(i) = 1 / (1 + (e^-z(i)));
endfor


% =============================================================

end
