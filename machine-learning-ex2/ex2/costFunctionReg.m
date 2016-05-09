function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

lambda_eye = eye(size(theta)(1));
lambda_eye(1, 1) = 0;
lambda_theta = lambda_eye * theta;

J = ((-1 * y)' * log(sigmoid(X * theta)) - (1 - y)' * log(1 - (sigmoid(X * theta)))) / m ...
    + (lambda_theta' * lambda_theta) * lambda / (2 * m);

grad = ((sigmoid(X * theta) - y)' * X)' / m + lambda_theta * lambda / m;

% =============================================================

end
