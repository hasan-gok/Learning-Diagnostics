function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

J = sum((X * theta - y).^2) /(2 * m) + lambda * sum(theta(2:end).^2)/(2 * m);

grad_reg = ones(size(theta')) .* theta' * lambda / m;
grad_reg(1) = 0;
grad = sum((X * theta - y) .* X,1)/m + grad_reg;
grad = grad(:);

end
