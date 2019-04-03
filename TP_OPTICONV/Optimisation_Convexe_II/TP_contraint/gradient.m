function grad_J = gradient(A,b,x)
% fonction J de la forme J(x) = ||Ax-b||²
grad_J = zeros(size(A,1),1);
for k = 1:size(A,1)
    grad_J(k) =  dot(2*A(:,k)',A*x-b);
end

    