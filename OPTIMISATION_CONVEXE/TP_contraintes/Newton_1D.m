function t = Newton_1D(x, d, t0, n_max, eps)

n_iter = 1;

%Calcul de la dérivée première et de la dérivée seconde
df = df_1D(x,d,t0);
df2 = d2f_1D(x,d,t0);

%Première itération pour trouver t
t = t0 - df/df2;

%On créé la boucle qui tourne tant que la distance entre t et t0 est sup à
%eps et que le nombre d'itération est inf à n_max

while(abs(t - t0) >= eps && n_iter <= n_max)
    
    t0 = t;
    df = df_1D(x,d,t0);
    df2 = d2f_1D(x,d,t0);
    
    if df2 == 0
        break;
    else
        t = t0 - df/df2;
        %if banane_1D(x,d,t) > banane_1D(x,d,t0)
          %  t = t0 + df/df2;
        %end
    
    
    end
    n_iter = n_iter + 1;
end

f = active_1D(x,d,t);

%fprintf('Le nombre ditération est de %d \n', n_iter);
%fprintf('Le minimum est atteint en t = %d et la valeur de la fonction en ce point est de %d',t,f);

end



