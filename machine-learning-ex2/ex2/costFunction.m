function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
temp0=0;temp1=0;temp2=0;
% ====================== YOUR CODE HERE ======================
for i=1:m
    z=transpose(theta)*transpose(X(i,:));
    hx=1/(1+exp(-z));
    temp0=temp0+(hx-y(i))*X(i,1);
    temp1=temp1+(hx-y(i))*X(i,2);
    temp2=temp2+(hx-y(i))*X(i,3);
    J=J+((y(i)*log(hx)) + ((1-y(i))*log(1-hx)));
end
J=J*(-1/m);
grad(1)=temp0/m;
grad(2)=temp1/m;
grad(3)=temp2/m;

    
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
