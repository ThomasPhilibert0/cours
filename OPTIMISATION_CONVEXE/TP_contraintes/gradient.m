function grad_J = gradient(A,b,x)
% fonction J de la forme J(x) = ||Ax-b||²
grad_J =  2*(A'*A*x - A'*b);


    