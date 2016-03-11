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



Xsize = size(X);
rows = Xsize(1);

J = (sum((log(sigmoid(X * theta)) .* -y) - (log(1 - sigmoid(X * theta)) .* (1- y)))/rows) + (lambda/ (2 * rows)) * sum( theta(2:1:end) .^2);
% the theta(2:1:end) return the vector without the first row

grad2 = (sum(diag(sigmoid(X * theta) - y) * X) / rows)';

grad =  grad2 + (lambda / rows) * theta;


grad(1) = grad2(1);



% =============================================================

end
