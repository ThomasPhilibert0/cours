function out = grad_opti(x0, n_max, eps)

X1(1) = x0(1);
X2(1) = x0(2);
n_iter = 1;

rho = Newton_1D(x0,JAC(x0),0,20,10e-5);

x = x0 + rho*JAC(x0);

while norm(x0 - x) > eps && n_iter <= n_max
    X1(n_iter + 1) = x(1);
    X2(n_iter + 1) = x(2);

    x0 = x;
    rho = Newton_1D(x0, JAC(x0), 0, 20, 10e-5);
    
    x = x0 + rho*JAC(x0);
    
    n_iter = n_iter +1;
end

plot(X1,X2)

f = banane(x);

fprintf('Le nombre ditération est de %d \n', n_iter);
fprintf('Le minimum est atteint en x = [%d,%d] et la valeur de la fonction en ce point est de %d',x(1),x(2),f);