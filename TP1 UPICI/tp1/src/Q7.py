import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import methodesQ7
import equations


def EQDiff() :
    """Résolution approchée de y' = A*y avec y(0) = (1,1) pour t dans [0,1] avec la méthode d'Euler"""

    A = np.diag([-1,-2])
    t0,y0 = 0.,np.array([1,1])
    T = 1.
    h = 0.2
    
    
    N = int(T/h)  
    [t,y1] = methodesQ7.saute_mouton(t0,h,N,y0,equations.f_diff,2,A)
 
    
    
    return y1