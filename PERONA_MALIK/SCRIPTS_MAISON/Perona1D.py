import numpy as np
import bord
import C1D
import math
from matplotlib import pyplot as plt

def Perona1D(u, dt, T, a):

    ''' Renvoie un comparatif entre un vecteur 1D et son traitement par Perona-Malik. On entrera le vecteur u, un pas de temps dt, un nombre d'itérations temporelles T et un parametre a pour la fonction C1D '''
    
    n = np.size(u,0)

    u_n = bord.bord1D(u, dt)
    u_n = u_n.astype(float)

    v_n = 0*u_n
    c_n = 0*u_n 

    plt.grid(True)
    plt.plot(u,'b',linewidth = 1.5,label = "Courbe exacte")
    plt.figure(1)
    
    for t in range(1,T):
        norm = C1D.ngrad1D(u_n)
        c_n = C1D.C1D(norm, a)

        #utilisation de la formule de discrétisation par DF
        for i in range(1, n-1):
            v_n[i] = u_n[i] + dt*(c_n[i]*(u_n[i+1]-u_n[i])-c_n[i-1]*(u_n[i]-u_n[i-1]))
        #On refait les bords pour chaque itération temporelle
        u_n = bord.bord1D(v_n, dt)
        
        if (t%(T/10) == 0):
                plotlabel = "t = %1.2f" %(t*dt)
                plt.plot(u_n, label=plotlabel, color = plt.get_cmap('copper')(float(t)/T))

    plt.title('Evolution Perona-Malik')
    plt.legend()
    
    plt.figure(2)
    plt.legend()
    plt.subplot(1,2,1)
    plt.plot(u,'b',linewidth = 1.5,label = "Courbe exacte")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(u_n,'r',linewidth = 1.5,label = "Courbe Perona-Malik")
    plt.legend()
    plt.suptitle("Comparatif Solution Exacte & Approchée")
    plt.show()
