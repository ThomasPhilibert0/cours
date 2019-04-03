function out = affichage(n_max, eps)

Met = input('Quel affichage souhaitez-vous ? fonction_1D / Newton_1D / Newton_2D /Gradient_fixe / Gradient_opti / Gradient_conj / Relaxation\n','s')


if (strcmp(Met, 'fonction_1D') || strcmp(Met, 'Newton_1D') || strcmp(Met, 'Newton_2D') || strcmp(Met, 'Gradient_fixe') || strcmp(Met, 'Gradient_opti') || strcmp(Met, 'Gradient_conj') || strcmp(Met, 'Relaxation') )
    
    if strcmp(Met,'fonction_1D')
        x = input('Vecteur x \n')
        d = input('Vecteur d \n')
        t = input('Linspace de t \n')
        
        for i = 1:length(t);
            T(i) = banane_1D(x,d,t(i));
        end
        
    plot(t,T)
    end
    
    
    if strcmp(Met,'Newton_1D')
        x = input('Vecteur x \n')
        d = input('Vecteur d \n')
        t0 = input('Linspace de t0 \n')
        
        [t,T] = Newton_1D(x,d,t0,n_max,eps);
        
        IT = [1:1:length(T)];
        
        plot(IT,T);
    end
    
    if strcmp(Met,'Newton_2D')
    end
    
    if strcmp(Met,'Gradient_fixe')
    end
    
    if strcmp(Met,'Gradient_opti')
    end
    
    if strcmp(Met,'Gradient_conj')
    end
    
    if strcmp(Met,'Relaxation')
    end
    
else
    disp('Voulez-vous dire : fonction_1D / Newton_1D / Newton_2D /Gradient_fixe / Gradient_opti / Gradient_conj / Relaxation ?')
end