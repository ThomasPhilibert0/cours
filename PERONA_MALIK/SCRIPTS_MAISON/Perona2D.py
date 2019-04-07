import numpy as np
import bord
import C2D
import math
from matplotlib import pyplot as plt


def Perona(u, dt, T, a):

    ''' Renvoie un comparatif entre une image 2D et son traitement par Perona-Malik. On entrera l'image discrétisée u, un pas de temps dt, un nombre d'itérations temporelles T et un parametre a pour la fonction C2D '''
    
    m = np.size(u,0)
    n = np.size(u,1)

    d = 2
    
    #code pour l'equation de Perona-Malik 2D
    u_n = bord.bord2D(u,d)
    u_n = u_n.astype(float)
    
    v_n = 0*u_n
    c_n = 0*u_n 
    w_n = np.zeros((m,n))

    #Boucle temporelle
    for k in range (0,T):
        norm = C2D.ngrad(u_n)
        c_n = C2D.C(norm,a)

        #Discrétisation par DF
        for i in range(d,m+d):
            for j in range(d,n+d):
                v_n[i][j] = u_n[i][j] + dt*(c_n[i][j]*(u_n[i+1][j]-u_n[i][j])-c_n[i-1][j]*(u_n[i][j]-u_n[i-1][j])+c_n[i][j]*(u_n[i][j+1]-u_n[i][j])-c_n[i][j-1]*(u_n[i][j]-u_n[i][j-1]))

        #Incrémentation des conditions de bord à chaque itération
        w_n = v_n[d:m+d, d:n+d]
        u_n = bord.bord2D(w_n,d)

    #Passage de l'image étendue à l'image de taille initiale
    w_n = v_n[d:m+d, d:n+d]

    plt.subplot(1,2,1)
    plt.imshow(u, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(w_n, cmap='gray')
    plt.show()
