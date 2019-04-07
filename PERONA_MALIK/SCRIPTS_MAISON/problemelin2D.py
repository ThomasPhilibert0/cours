import numpy as np
import bord
import math
from matplotlib import pyplot as plt


def lineaire2D(u, dt, T):

    
    ''' Renvoie un comparatif entre une image 2D et son traitement par l'équation de la chaleur linéaire. On entrera l'image discrétisée u, un pas de temps dt, un nombre d'itérations temporelles T'''
    
    n = np.size(u,0)
    m = np.size(u,1)

    d = 1
    
    #code pour l'equation de la chaleur 2D
    u_n = bord.bord2D(u,d)
    v_n = 0*u_n
    w_n = np.zeros((n,m))

    #Itérations temporelles
    for k in range (0,T):
        for i in range(d,n+d):
            for j in range(d,m+d):
                #Discrétisation par DF
                v_n[i][j] = u_n[i][j] + 0.5*dt * (u_n[i+1][j] + u_n[i-1][j] - 4*u_n[i][j] + u_n[i][j+1] + u_n[i][j-1])      
        #On refait les conditions de bord à chaque itération
        w_n = v_n[d:n+d, d:m+d]
        u_n = bord.bord2D(w_n,d)

    plt.subplot(1,2,1)
    plt.imshow(u, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(w_n, cmap='gray')
    plt.show()
    


