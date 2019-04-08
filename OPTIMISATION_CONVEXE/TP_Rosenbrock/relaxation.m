function out = relaxation(x0, n_max, eps)

n_iter = 1;

e1 = [1;0];
e2 = [0;1];

X1 = x0(1);
X2 = x0(2);

x = x0;
x(1) = x0(1) + Newton_1D(x0, e1, 0, 100, eps);
x0 = x;
x(2) = x0(2) + Newton_1D(x0, e2, 0, 100, eps);


while norm(x-x0) >= eps && n_iter < n_max
    X1(n_iter +1) = x(1);
    X2(n_iter +1) = x(2);
    
    x0 = x;
    x(1) = x0(1) + Newton_1D(x0, e1, 0, 100, eps);
    x0 = x;
    x(2) = x0(2) + Newton_1D(x0, e2, 0, 100, eps);
    
    n_iter = n_iter + 1;
end

plot(X1,X2);
f = banane(x);

fprintf('Le nombre ditération est de %d \n', n_iter)
fprintf('Le minimum est atteint en x = [%d,%d] et la valeur de la fonction en ce point est de %d',x(1),x(2),f)

    
    