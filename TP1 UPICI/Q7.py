import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import methodesQ7
import equations


def EQDiff() :
    """Résolution approchée de y' = A*y avec y(0) = (1,1) pour t dans [0,1] """

    A = np.diag([-1,-2])
    t0,y0 = 0.,np.array([1,1])
    T = 1.
    h = 0.2
    m = np.size(A)//2
    
    N = int(T/h)  
    [t,y1] = methodesQ7.euler_explicite(t0,h,N,y0,equations.f_diff,2,A)
    [t,y2] = methodesQ7.RK2(t0,h,N,y0,equations.f_diff,2,A)
    [t,y3] = methodesQ7.saute_mouton(t0,h,N,y0,equations.f_diff,2,A)
    [t,y4] = methodesQ7.trapezePC(t0,h,N,y0,equations.f_diff,2,A)
    
    t = np.linspace(t0,t0+N*h,N+1)
    z1 = np.zeros([m,np.size(t)])
    y = np.zeros([m,np.size(t)])
    y[:,0] = y0
    C = np.zeros(m)    
    for k in np.arange(np.size(t)):
        for i in np.arange(m):
            C = equations.f_exact(t[k], A, y0)
            z1[i][k] = C[i]
    
    return y1,y2,y3,y4,z1