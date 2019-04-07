import numpy as np
import bord
import math
from matplotlib import pyplot as plt

def lineaire1D(u, dt, T):

    '''Renvoie un comparatif entre un vecteur 1D et son traitement par l'équation de la chaleur linéaire. On entrera le vecteur u, un pas de temps dt, un nombre d'itérations temporelles T'''

    N = np.size(u,0)
    
    #condition de cfl, on prendra un dx toujours = 1. On a ainsi la stabilité.
    cfl = 0.25*dt

    #conditions de bord 1D
    u_n = bord.bord1D(u, dt)
    v_n = 0*u_n

    plt.grid(True)
    plt.plot(u,'b',linewidth = 1.5,label = "Courbe exacte")
    
    # code pour l'equation de la chaleur 1D
    for t in range (1,T):
            for i in range(1, N-1):
                  #Discrétisatoin par DF
                  v_n[i] = u_n[i] + cfl * (u_n[i-1]-2.*u_n[i]+u_n[i+1])

            u_n = bord.bord1D(v_n, dt)
            
            if (t%(T/10) == 0):
                plotlabel = "t = %1.2f" %(t*dt)
                plt.plot(u_n, label=plotlabel, color = plt.get_cmap('copper')(float(t)/T))
                        

    plt.title('Equation de la chaleur 1D')
    plt.legend()
    plt.show()
