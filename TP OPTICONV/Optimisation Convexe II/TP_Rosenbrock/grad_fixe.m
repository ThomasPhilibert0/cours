function out = grad_fixe(x0, rho, n_max, eps)

%On initialise le nombre d'itérations
n_iter = 1;

%On donne la première valeur de x par rapport à la direction -rho*JAC(x0)
x = direction(x0,rho);

%On fait la boucle 
while norm(x - x0) >= eps && n_iter <= n_max
    x0 = x;
    x = direction(x0,rho);
    n_iter = n_iter + 1;
end

f = banane(x);

%fprintf('Le nombre ditération est de %d \n', n_iter)
%fprintf('Le minimum est atteint en x = [%d,%d] et la valeur de la fonction en ce point est de %d',x(1),x(2),f)

    
    