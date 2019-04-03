function out = grad_opti(x0, n_max, eps)

n_iter = 1;
GRAD = JAC(x0);
rho = abs(Newton_1D(x0, GRAD, -1, 10, eps));
x = direction(x0,rho);

while norm(x-x0) >= eps && n_iter <= n_max
    x0 = x;
    GRAD = JAC(x0);
    rho = abs(Newton_1D(x0, GRAD, -1, 10, eps));
    x = direction(x0,rho);
    n_iter = n_iter +1;
end
f = banane(x);

fprintf('Le nombre ditération est de %d \n', n_iter);
fprintf('Le minimum est atteint en x = [%d,%d] et la valeur de la fonction en ce point est de %d',x(1),x(2),f);