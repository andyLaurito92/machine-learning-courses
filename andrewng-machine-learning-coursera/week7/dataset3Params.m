function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_values = C_values;

C_min_error = 0;
sigma_min_error = 0;
min_error = 3000;

for C_i = C_values;
  for sigma_i = sigma_values;
    model = svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_i));
    predictions = svmPredict(model, Xval);
    current_error = mean(double(predictions ~= yval));
    if current_error < min_error;
      C_min_error = C_i;
      sigma_min_error = sigma_i;
      min_error = current_error;
    endif
  endfor
endfor

C = C_min_error;
sigma = sigma_min_error;



% =========================================================================

end
