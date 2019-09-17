import numpy as np
import scipy.sparse as sparse   # Algèbre linéaire creuse
import matplotlib.pyplot as plt # Pour les graphiques
import scipy.sparse.linalg as sci
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import bmat
from math import pi
from dom import Dom_init
from dom import masque
from dist import penalisation


def skeleton(N,R,ray_tub) :

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    SKELET = np.loadtxt("penal_1000_10^-7_ouvert")
    A = np.zeros((N+1,N+1))


    ##Codage pour P4
    for j in range(int(N/2+R*(N+1))):
        MAX = max(SKELET[j,:])
        for i in range(N+1):
            if SKELET[j][i] != MAX :
                A[j][i] = 0
            else:
                A[j][i] = SKELET[j][i]

    ##Codage pour P1
    for i in range(int(N/2)):
        MAX = max(SKELET[int(N/2+R*(N+1)+1):,i])
        for j in range(int(N/2+R*(N+1)+1),N+1):
            if SKELET[j][i] != MAX :
                A[j][i] = 0
            else:
                A[j][i] = SKELET[j][i]

    ##Codage pour P2
    for i in range(int(N/2),N+1):
        MAX = max(SKELET[int(N/2+R*(N+1)+1):,i])
        for j in range(int(N/2+R*(N+1)+1),N+1):
            if SKELET[j][i] != MAX :
                A[j][i] = 0
            else:
                A[j][i] = SKELET[j][i]

    ##Codage pour P3
    for j in range(int(N/2+R*(N+1)),N+1):
        MAX = max(SKELET[j,int(N/2-ray_tub/3):int(N/2+ray_tub/3)])
        for i in range(int(N/2)-ray_tub,int(N/2)+ray_tub):
            if SKELET[j][i] != MAX :
                A[j][i] = 0
            else:
                A[j][i] = SKELET[j][i]
    
    

    
                
    fig = plt.figure(figsize = plt.figaspect(0.35))

    ax = fig.add_subplot(111, projection = '3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,A, cmap = 'hot')
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

    return A
