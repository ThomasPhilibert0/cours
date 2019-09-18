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


def skeleton(N) :

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    dir1 = 0
    dir2 = 0
    dir2 = 0
    dir3 = 0
    dir4 = 0

    MAT = np.loadtxt("penal_20_6")
    SKELET = np.zeros((N+1,N+1))
    DIR = 0

    for i in range(1,N):
        for j in range(1,N):
            pix = MAT[j][i]
            #Première direction(verticale)
            if pix > MAT[j-1][i] and pix > MAT[j+1][i]:
                dir1 = 1
                DIR = dir1
            #Seconde direction(horizontale)
            if pix > MAT[j][i+1] and pix > MAT[j][i-1]:
                dir2 = 1
                DIR = DIR + dir2
            #Troisieme direction(diagonale g_d)
            if pix > MAT[j-1][i+1] and pix > MAT[j+1][i-1]:
                dir3 = 1
                DIR = DIR + dir3
            #Quatrième direction(diagonale d_g)
            if pix > MAT[j-1][i-1] and pix > MAT[j+1][i+1]:
                dir4 = 1
                DIR = DIR + dir4

        SKELET[j][i] = DIR
            
    fig = plt.figure(figsize = plt.figaspect(0.35))

    ax = fig.add_subplot(111, projection = '3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,SKELET, cmap = 'hot')
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

    return SKELET
