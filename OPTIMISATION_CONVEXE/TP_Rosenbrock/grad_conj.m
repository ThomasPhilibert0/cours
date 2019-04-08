function out = grad_conj(x0,n_max,eps)

X1(1) = x0(1);
X2(1) = x0(2);

n_iter = 1;

d0 = JAC(x0);

if d0 == 0
    x = x0;

else
    r0 = Newton_1D(x0, d0, -1, 100, eps);
    x = x0 + d0*r0;
    
    while norm(x-x0) > eps && n_iter < n_max
        X1(n_iter + 1) = x(1);
        X2(n_iter + 1) = x(2);
        
        d = JAC(x) + (dot(JAC(x) , JAC(x) - JAC(x0)) / norm(JAC(x0))^2)*d0;
        
        if d == 0
            break;
        
        else
            r0 = Newton_1D(x,d,-1,100,eps);
            
            x0 = x;
            d0 = d;
            
            x = x0 + d0*r0;
        end
        
        n_iter = n_iter + 1;
    end
end

plot(X1,X2)

f = banane(x);

fprintf('Le nombre ditération est de %d \n', n_iter)
fprintf('Le minimum est atteint en x = [%d,%d] et la valeur de la fonction en ce point est de %d',x(1),x(2),f)