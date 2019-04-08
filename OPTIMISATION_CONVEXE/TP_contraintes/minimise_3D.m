function [x,t] = minimise_3D(A,b,tau,r)

% point d'initialisation
x0 = [4; 4; 8]; 

% L = 2*norm(A'A) = 2 * pgvp A'A
% alpha = 2*ppvp A'A 

[~,valeurs_propres]=eig(A'*A);
L = 2*max(max(valeurs_propres));
alpha = 2*valeurs_propres(1,1);

n_iter = 1;

t = 2*alpha/L^2 ;

while n_iter <= r
    
    if tau >= 2*alpha/L^2 
        disp('Erreur: tau trop grand');
        break
    end
    
    x = projection_3D(x0 - tau*gradient(A,b,x0));
    x0 = x;
    n_iter = n_iter+1;
    
end