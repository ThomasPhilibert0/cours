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
import time


def convolution(DIST,Noyau):
    
    N = np.shape(DIST)[0] - 1
    M = np.shape(Noyau)[0] - 1
    L = int(M/2)+1    
    
    CONV = np.zeros((N+1,N+1))

    for i in range (L,N-L):
        for j in range(L,N-L):
            a = 0
            for k in range(-L,L-1):
                for l in range(-L,L-1):
                    a += DIST[i+k][j+l]*Noyau[L+k][L+l]
            CONV[i][j]= a
                    
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)
    
    fig = plt.figure(figsize = plt.figaspect(0.35))
    ax = fig.add_subplot(111, projection = '3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,CONV, cmap = 'hot')
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()
    
    return CONV