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
%+ lambda/(2*m)*(dot(theta,theta) - theta(1)*thata(1))


%Use theta_changed to make a new vetcor whose theta(1) is 0, others all the
%same
theta_changed = [0; theta(2:end)];
%Use the theta_changed in avoid of changing theta(1)
J = (1/m) * (-y' * log(sigmoid(X * theta)) - (1 - y') * log(1 - sigmoid(X * theta)))+ lambda/(2*m)*(dot(theta_changed,theta_changed));

%Similarly as before
grad = (1/m) * X' * (sigmoid(X * theta) - y) + lambda / m * theta_changed;




% =============================================================

end
