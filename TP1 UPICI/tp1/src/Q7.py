import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import methodesQ7
import equations


def EQDiff() :
    """Résolution approchée de y' = A*y avec y(0) = (1,1) pour t dans [0,1] avec la méthode d'Euler"""

    equations.A = np.diag([-1,-2])
    t0,y0 = 0.,np.array([1,1])
    T = 1.
    h = 0.2
    
    print("Valeur de l'erreur en fonction de h")
    
    #Définition de la boucle for
    for i in range(0,6,1) :
        N = int(T/h) #Nombre d'itérations
        
        [t,y1] = methodesQ7.euler_explicite(t0,h,N,y0,equations.f_diff,2)
        
        
        h = h/2
       
    return y1