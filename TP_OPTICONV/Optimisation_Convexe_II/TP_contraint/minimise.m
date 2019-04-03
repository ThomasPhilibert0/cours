function x = minimise(A,b,tau,r)
x0 = [0; 1]; % point d'initialisation
i = 1;
while i <= r
    x = projection(x0 - tau*gradient(A,b,x0));
    x0 = x;
    i = i+1;
end

    