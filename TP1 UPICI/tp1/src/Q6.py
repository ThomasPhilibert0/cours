import numpy as np
import matplotlib.pyplot as plt

import methodes
import equations

def comptrig1() :
    
    """ Comparaison graphe convergence de y' = a*y(t) + (1-a)*cos(t) -(1-a)*sin(t)  avec y(0) = 1  pour t dans [0,5] avec les 4 méthodes et a = -1"""

    equations.a = -1.
    t0,y0 = 0.,1.
    T = 5.
    h = np.array([0.5,0.1,0.01])
    
    #Définition de la boucle for
    for i in range(0,3,1) :
        N = int(T/h[i])  #Nombre d'itérations

        [t,y1] = methodes.euler_explicite(t0,h[i],N,y0,equations.f_trig)
        [t,y2] = methodes.RK2(t0,h[i],N,y0,equations.f_trig)
        [t,y3] = methodes.saute_mouton(t0,h[i],N,y0,equations.f_trig)
        [t,y4] = methodes.trapezePC(t0,h[i],N,y0,equations.f_trig)
        
        # Solution exacte aux mêmes instants
        z1 = equations.sol_trig(t)
        
        #Graphe des solutions exactes et approchées
        plt.figure(i)
        plt.plot(t,y1,'b-+')
        plt.plot(t,y2,'r-+')
        plt.plot(t,y3,'g-+')
        plt.plot(t,y4,'m-+')
        plt.plot(t,z1,'k')
        plt.xlabel('t')
        plt.ylabel('y')
        plt.title("Comparatif des différentes méthodes pour h =" +str(h[i]))

        # Calcul de l'erreur maximum relative
        e1 = np.max(np.abs((z1-y1)/z1))
        e2 = np.max(np.abs((z1-y2)/z1))
        e3 = np.max(np.abs((z1-y3)/z1))
        e4 = np.max(np.abs((z1-y4)/z1))

        plt.figure(3)
        plt.plot(h[i],e1,'b-+',label = 'Méthode Euler explicite')
        plt.legend()
        plt.plot(h[i],e2,'r-+', label = 'Méthode du RK2')
        plt.legend()
        plt.plot(h[i],e3,'g-+', label = 'Méthode saute-mouton')
        plt.legend()
        plt.plot(h[i],e4,'m-+', label = 'Méthode trapèze PC')
        plt.legend()
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('h')
    plt.ylabel('e1,e2,e3,e4')
    plt.title("Erreur en fonction du pas h")
    plt.show()
    
def comptrig10() :
    
    """ Comparaison graphe convergence de y' = a*y(t) + (1-a)*cos(t) -(1-a)*sin(t)  avec y(0) = 1  pour t dans [0,5] avec les 4 méthodes et a = -10"""

    equations.a = -10.
    t0,y0 = 0.,1.
    T = 5.
    h = np.array([0.5,0.1,0.01])
    
    #Définition de la boucle for
    for i in range(0,3,1) :
        N = int(T/h[i])  #Nombre d'itérations

        [t,y1] = methodes.euler_explicite(t0,h[i],N,y0,equations.f_trig)
        [t,y2] = methodes.RK2(t0,h[i],N,y0,equations.f_trig)
        [t,y3] = methodes.saute_mouton(t0,h[i],N,y0,equations.f_trig)
        [t,y4] = methodes.trapezePC(t0,h[i],N,y0,equations.f_trig)
        
        # Solution exacte aux mêmes instants
        z1 = equations.sol_trig(t)

        #Graphe des solutions exactes et approchées
        plt.figure(i)
        plt.plot(t,y1,'b-+')
        plt.plot(t,y2,'r-+')
        plt.plot(t,y3,'g-+')
        plt.plot(t,y4,'m-+')
        plt.plot(t,z1,'o')
        plt.xlabel('t')
        plt.ylabel('y')
        plt.title("Comparatif des différentes méthodes pour h =" + str(h[i]))

        # Calcul de l'erreur maximum relative
        e1 = np.max(np.abs((z1-y1)/z1))
        e2 = np.max(np.abs((z1-y2)/z1))
        e3 = np.max(np.abs((z1-y3)/z1))
        e4 = np.max(np.abs((z1-y4)/z1))

        plt.figure(3)
        plt.plot(h[i],e1,'b-+')
        plt.plot(h[i],e2,'r-+')
        plt.plot(h[i],e3,'g-+')
        plt.plot(h[i],e4,'m-+')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('h')
    plt.ylabel('e1,e2,e3,e4')
    plt.title("Erreur en fonction du pas h")
    plt.legend()
    plt.show()

def comptrig50() :
    
    """ Comparaison graphe convergence de y' = a*y(t) + (1-a)*cos(t) -(1-a)*sin(t)  avec y(0) = 1  pour t dans [0,5] avec les 4 méthodes et a = -50"""

    equations.a = -50.
    t0,y0 = 0.,1.
    T = 5.
    h = np.array([0.5,0.1,0.01])
    
    #Définition de la boucle for
    for i in range(0,3,1) :
        N = int(T/h[i])  #Nombre d'itérations

        [t,y1] = methodes.euler_explicite(t0,h[i],N,y0,equations.f_trig)
        [t,y2] = methodes.RK2(t0,h[i],N,y0,equations.f_trig)
        [t,y3] = methodes.saute_mouton(t0,h[i],N,y0,equations.f_trig)
        [t,y4] = methodes.trapezePC(t0,h[i],N,y0,equations.f_trig)
        
        # Solution exacte aux mêmes instants
        z1 = equations.sol_trig(t)

        #Graphe des solutions exactes et approchées
        plt.figure(i)
        plt.plot(t,y1,'b-+')
        plt.plot(t,y2,'r-+')
        plt.plot(t,y3,'g-+')
        plt.plot(t,y4,'m-+')
        plt.plot(t,z1,'o')
        plt.title("Comparatif des différentes méthodes pour h = " +str(h[i]))

        # Calcul de l'erreur maximum relative
        e1 = np.max(np.abs((z1-y1)/z1))
        e2 = np.max(np.abs((z1-y2)/z1))
        e3 = np.max(np.abs((z1-y3)/z1))
        e4 = np.max(np.abs((z1-y4)/z1))

        plt.figure(3)
        plt.plot(h[i],e1,'b-+')
        plt.plot(h[i],e2,'r-+')
        plt.plot(h[i],e3,'g-+')
        plt.plot(h[i],e4,'m-+')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('h')
    plt.ylabel('e1,e2,e3,e4')
    plt.title("Erreur en fonction du pas h")
    plt.legend()
    plt.show()


