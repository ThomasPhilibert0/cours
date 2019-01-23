import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.spatial import distance
import methodesQ7
import equations


def EQDiff() :
    """Résolution approchée de y' = A*y avec y(0) = (1,1) pour t dans [0,1] """

    A = np.diag([-1,-2])
    t0,y0 = 0.,np.array([1,1])
    T = 1.
    h = 0.2
    m = np.size(A)//2
    f1 = np.zeros(6)
    f2 = np.zeros(6)
    f3 = np.zeros(6)
    f4 = np.zeros(6)
    
    
    for l in range(0,6,1) :
        N = int(T/h) 
        t = np.linspace(t0,t0+N*h,N+1)
        e1 = np.zeros([np.size(t)])
        e2 = np.zeros([np.size(t)])
        e3 = np.zeros([np.size(t)])
        e4 = np.zeros([np.size(t)])
        
        [t,y1] = methodesQ7.euler_explicite(t0,h,N,y0,equations.f_diff,2,A)
        [t,y2] = methodesQ7.RK2(t0,h,N,y0,equations.f_diff,2,A)
        [t,y3] = methodesQ7.saute_mouton(t0,h,N,y0,equations.f_diff,2,A)
        [t,y4] = methodesQ7.trapezePC(t0,h,N,y0,equations.f_diff,2,A)
        
        
        z1 = np.zeros([m,np.size(t)])
        y = np.zeros([m,np.size(t)])
        y[:,0] = y0
        C = np.zeros(m)    
        for k in np.arange(np.size(t)):
            for i in np.arange(m):
                C = equations.f_exact(t[k], A, y0)
                z1[i][k] = C[i]
        
        for k in np.arange(np.size(t)):
            e1[k] = distance.euclidean(z1.T[k],y1.T[k])
            e2[k] = distance.euclidean(z1.T[k],y2.T[k])
            e3[k] = distance.euclidean(z1.T[k],y3.T[k])
            e4[k] = distance.euclidean(z1.T[k],y4.T[k])
            
        f1[l]=max(e1)
        f2[l]=max(e2)
        f3[l]=max(e3)
        f4[l]=max(e4)
        plt.plot(h,f1[l],'b-+',label = 'Pour h = '+ str(h))
        
        h=h/2
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('h')
    plt.ylabel('e1')
    plt.title("Erreur en fonction du pas h")
    plt.legend()
    plt.show()
    
    return f1