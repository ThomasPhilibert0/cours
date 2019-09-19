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


def skeleton(N,lim) :

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    DIR = 0
    dir1 = 0
    dir2 = 0
    dir2 = 0
    dir3 = 0
    dir4 = 0

    MAT = np.loadtxt("fast_marching_3000")
    SKELET = np.zeros((N+1,N+1))
    

    for i in range(1,N):
        for j in range(1,N):
            pix = MAT[i][j]

            #Première direction(verticale)
            if pix > MAT[i-1][j] and pix > MAT[i+1][j]:
                dir1 = 1
            #Seconde direction(horizontale)
            if pix > MAT[i][j-1] and pix > MAT[i][j+1]:
                dir2 = 1
            #Troisieme direction(diagonale g_d)
            if pix > MAT[i-1][j-1] and pix > MAT[i+1][j+1]:
                dir3 = 1
            #Quatrième direction(diagonale d_g)
            if pix > MAT[i+1][j-1] and pix > MAT[i-1][j+1]:
                dir4 = 1

            DIR = dir1 + dir2 + dir3 + dir4

            if DIR >= lim :
                SKELET[i][j] = DIR

            DIR = 0
            dir1 = 0
            dir2 = 0
            dir2 = 0
            dir3 = 0
            dir4 = 0    

    fig = plt.figure(figsize = plt.figaspect(0.35))

    ax = fig.add_subplot(111, projection = '3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,MAT, cmap = 'hot')
    plt.xlabel("x")
    plt.ylabel("y")

    fig2 = plt.figure(figsize = plt.figaspect(0.35))

    ax = fig2.add_subplot(111)
    X,Y = np.meshgrid(x,y)
    ax.contourf(X,Y,SKELET, cmap = 'binary')
    #ax.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

    return SKELET
