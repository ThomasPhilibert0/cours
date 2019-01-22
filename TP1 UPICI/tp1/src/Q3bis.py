import numpy as np
import matplotlib.pyplot as plt

import methodes
import equations

def compeq1() :
    
    """ Comparaison graphe convergence de y' = 1-y avec y(0) = 5 pour t dans [0,1] avec méthode d'Euler explicite et RK2"""

    equations.a = -1.
    equations.b = 1.
    t0,y0 = 0.,5.
    T = 1.
    h = 0.2

    #Définition de la boucle for
    for i in range(0,6,1) :
        N = int(T/h) #Nombre d'itérations

        [t,y1] = methodes.euler_explicite(t0,h,N,y0,equations.f_affine)
        [t,y2] = methodes.RK2(t0,h,N,y0,equations.f_affine)

        # Solution exacte aux mêmes instants
        z1 = equations.sol_affine(t,y0)

        # Calcul de l'erreur maximum relative
        e1 = np.max(np.abs((z1-y1)/z1))
        e2 = np.max(np.abs((z1-y2)/z1))
        
        plt.plot(h,e1,'b-+')
        plt.plot(h,e2,'r-+')
        h = h/2

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('h')
    plt.ylabel('e1,e2')
    plt.title("Erreur en fonction du pas h")
    plt.legend()
    plt.show()


def compeq2():
    """Comparaison graphe de convergence y' = 1-y^2 avec y(0) = 0 pour t dans [0,1] avec la méthode d'Euler explicite et RK2"""

    equations.a = -1.
    equations.b = 1.
    t0,y0 = 0.,0.
    T = 1.
    h = 0.2

    #Définition de la boucle for
    for i in range(1,6,1) :
        N = int(T/h) #Nombre d'itérations

        [t,y1] = methodes.euler_explicite(t0,h,N,y0,equations.f_poly)
        [t,y2] = methodes.RK2(t0,h,N,y0,equations.f_poly)

        # Solution exacte aux mêmes instants
        z1 = equations.sol_poly1(t)

        # Calcul de l'erreur maximum relative
        e1 = np.nanmax(np.abs((z1-y1)/z1))
        e2 = np.nanmax(np.abs((z1-y2)/z1))

        plt.plot(h,e1,'b-+')
        plt.plot(h,e2,'r-+')

        h = h/2
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('h')
    plt.ylabel('e1')
    plt.title("Erreur en fonction du pas h")
    plt.legend()
    plt.show()

def compeq3():
    """Comparaison graphe de convergence y' = 1-y^2 avec y(0) = 2 pour t dans [0,1] avec la méthode d'Euler explicite et RK2"""

    equations.a = -1.
    equations.b = 1.
    t0,y0 = 0.,2.
    T = 1.
    h = 0.2

    #Définition de la boucle for
    for i in range(1,6,1) :
        N = int(T/h) #Nombre d'itérations

        [t,y1] = methodes.euler_explicite(t0,h,N,y0,equations.f_poly)
        [t,y2] = methodes.RK2(t0,h,N,y0,equations.f_poly)

        # Solution exacte aux mêmes instants
        z1 = equations.sol_poly2(t)

        # Calcul de l'erreur maximum relative
        e1 = np.nanmax(np.abs((z1-y1)/z1))
        e2 = np.nanmax(np.abs((z1-y2)/z1))

        plt.plot(h,e1,'b-+')
        plt.plot(h,e2,'r-+')

        h = h/2
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('h')
    plt.ylabel('e1')
    plt.title("Erreur en fonction du pas h")
    plt.legend()
    plt.show()
