import skfmm
import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from dom import masque
import time
import scipy as sc
from scipy import signal

def skeleton_laplace(LISS):

    N = np.shape(LISS)[0] - 1

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    MAT = np.zeros((N+1,N+1))
    
    dy,dx = np.gradient(LISS)

    dy2,dyx = np.gradient(dy)
    dxy,dx2 = np.gradient(dx)

    LAPLACE = np.abs(dy2 + dx2)

    for i in range(N+1):
        for j in range(N+1):
            if LAPLACE[i][j] >= 0.001:
                MAT[i][j] = 1

    
    fig = plt.figure(figsize = plt.figaspect(0.35))
    ax = fig.add_subplot(111)
    X,Y = np.meshgrid(x,y)
    ax.contour(X,Y,MAT, cmap = 'hot')
    plt.xlabel("x")
    plt.ylabel("y")

    fig2 = plt.figure(figsize = plt.figaspect(0.35))
    ax = fig2.add_subplot(111,projection = '3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,LAPLACE, cmap = 'hot')
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

    return LAPLACE

def skeleton_method(DIST):

    N = np.shape(DIST)[0] - 1

    h = 1/(N+1)

    MAT = np.zeros((N+1,N+1))
    
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    for i in range(1,N):
        for j in range(1,N):
            dx_droite = (DIST[i+1,j] - DIST[i,j])/h
            dx_gauche = (DIST[i-1,j] - DIST[i,j])/h
            dy_haut = (DIST[i,j-1] - DIST[i,j])/h
            dy_bas = (DIST[i,j+1] - DIST[i,j])/h

            d1 = dx_droite*dx_gauche
            d2 = dx_droite*dy_haut
            d3 = dx_droite*dy_bas
            d4 = dx_gauche*dy_haut
            d5 = dx_gauche*dy_bas
            d6 = dy_haut*dy_bas

            MIN = min(d1,d2,d3,d4,d5,d6)

            if MIN <= -0.001:
                MAT[i,j] = 1


    fig = plt.figure(figsize = plt.figaspect(0.35))
    ax = fig.add_subplot(111)
    X,Y = np.meshgrid(x,y)
    ax.contour(X,Y,MAT, cmap = 'hot')
    plt.xlabel("x")
    plt.ylabel("y")

    fig2 = plt.figure(figsize = plt.figaspect(0.35))
    ax = fig2.add_subplot(111,projection = '3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,MAT, cmap = 'hot')
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()      

    return MAT,MIN
