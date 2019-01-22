import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import méthodesQ7
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
        
        [t,y1] = méthodesQ7.euler_explicite(t0,h,N,y0,equations.f_diff,2)
        
        # Solution exacte aux mêmes instants
        z1 = equations.sol_diff(t,y0)
        
        # Calcul de l'erreur maximum relative
        e1 = np.max(np.abs((z1-y1)/z1))
        
        
        #Graphe des solutions exactes et approchées
        plt.figure(1)
        plt.plot(t,y1,'b-+')
        plt.plot(t,z1,'r')
        plt.xlabel('t')
        plt.ylabel('y')
        plt.title("Euler explicite")
        
        # Écriture de l'erreur en fonction de h
        print("{0} | {1}".format(h,e1))
        plt.figure(2)
        plt.plot(h,e1,'b-+',label = 'Pour h = '+ str(h))
        
        h = h/2
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('h')
    plt.ylabel('e1')
    plt.title("Erreur en fonction du pas h")
    plt.legend()
    plt.show()
