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

def skeleton(N, dt, ray_tub, R, Hg, Hd, Lb, angle):

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    SKELET = penalisation(N, dt, ray_tub, R, Hg, Hd, Lb, angle)
    A = np.zeros((N+1,N+1))
    epsilon = 1/(2*(N+1))
    
    for j in range(N+1):
        MAX = max(SKELET[j,:])
        for i in range(N+1):
            if SKELET[j][i] != MAX :
                A[j][i] = 0
            else :
                A[j][i] = MAX

    for i in range(N+1):
        if i < int((N+1)/2) - 1 or i > int((N+1)/2) +1:
            MAX = max(SKELET[int((N+1)/2)+int(R*N):,i])
            for j in range(int((N+1)/2)+int(R*N),N+1):
                if SKELET[j][i] != MAX :
                    A[j][i] = 0
                else :
                    A[j][i] = MAX


        
    
    fig = plt.figure(figsize = plt.figaspect(0.35))
    ax = fig.add_subplot(111, projection = '3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,A, cmap = 'hot')
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

    return SKELET
