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
    '''NOYAU = (1/256)*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])
    LAP = np.array([[0,1,0],[1,-4,1],[0,1,0]])'''
    
    N = np.shape(DIST)[0] - 1
    M = np.shape(Noyau)[0] - 1
    L = int(M/2)+1    
    
    CONV = np.zeros((N+1,N+1))

    for i in range (L,N-L+1):
        for j in range(L,N-L+1):
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

def skeleton(conv,eps):
    N = np.shape(conv)[0] - 1
    MAT = np.zeros((N+1,N+1))
    for i in range(N+1):
        for j in range(N+1):
            if conv[i][j]<eps:
                MAT[i][j]=1
                
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)
    
    fig = plt.figure(figsize = plt.figaspect(0.35))
    ax = fig.add_subplot(111)
    X,Y = np.meshgrid(x,y)
    ax.contour(X,Y,MAT, cmap = 'hot')
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()