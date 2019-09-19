import numpy as np
import scipy.sparse as sparse   # Algèbre linéaire creuse
import matplotlib.pyplot as plt # Pour les graphiques
import scipy.sparse.linalg as sci
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import bmat
from math import pi
from dom import Dom_init
from dom import masque

def matrix_lap2(N,dt):
    """Retourne une matrice qui discrétise le laplacien de u dans le domaine Omega = [xmin,xmax,ymin,ymax], découpé en N intervalles en x et y. La matrice finale est une matrice scipy.sparse CSR matrix. Cette matrice est de taille (N+1)*(N+1)"""

    h = 1./N
    h2 = h*h

    #On note les inconnues de 0 à Nx suivant x et 0 à Ny suivant y. La taille du problème est donc (Nx+1)*(Ny+1).

    #Cela correspond à x_i = i*h et y_j = j*h et la numérotation (i,j) --> k := (N+1)*j+i.

    taille = (1+N)*(1+N)

    diags = np.zeros((5,taille))

    #Diagonale principale
    diags[2,:] = 1.
    diags[2, N+2:taille - (N+2)] = 1 + ((4*dt)/h2)
    diags[2, np.arange(2*N+1, taille, N+1)] = 1.
    diags[2, np.arange(2*N+2, taille, N+1)] = 1.
              
    #Diagonale "-1"
    diags[1,N+1:taille-(N+1)] = -dt/h2
    diags[1, np.arange(2*N, taille, N+1)] = 0.
    diags[1, np.arange(2*N+1, taille, N+1)] = 0.
    
    #Diagonale "+1"
    diags[3, N+3:taille-(N+1)] = -dt/h2
    diags[3, np.arange(2*N+2, taille, N+1)] = 0.
    diags[3, np.arange(2*N+3, taille, N+1)] = 0.

    #Diagonale "-(N+1)"
    diags[0, 1 : taille - (2*N+3)] = -dt/h2
    diags[0, np.arange(N,taille,N+1)] = 0.
    diags[0, np.arange(N+1,taille,N+1)] = 0.

    #Diagonale "+(N+1)"
    diags[4, taille - N*N + 2 : taille - 1] = -dt/h2
    diags[4, np.arange(taille - N*N + 1 + N ,taille,N+1)] = 0.
    diags[4, np.arange(taille - N*N + 2 + N ,taille,N+1)] = 0.

    #Construction de la matrice creuse
    A = sparse.spdiags(diags,[-(N+1),-1,0,1,(N+1)],taille,taille, format = "csr")

    return A


def chaleurdist(N, dt, ray_tub, R, Hg, Hd, Lb, angle):

     x = np.linspace(0,1,N+1)
     y = np.linspace(0,1,N+1)

     taille = (N+1)*(N+1)

     T = np.ones(taille)
     MAT = Dom_init(N, ray_tub, R, Hg, Hd, Lb, angle)

     for i in range(1,N):
         for j in range(1,N):
             k = i*(N+1) + j
             T[k] = MAT[i][j]

     D = sci.spsolve(matrix_lap2(N,dt), T)
         
     return D

def dist(N, dt, ray_tub, R, Hg, Hd, Lb, angle):

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    T = chaleurdist(N, dt, ray_tub, R, Hg, Hd, Lb, angle)

    dist  = -np.log(T)*np.sqrt(dt)

    #print(np.max(dist))

    fig = plt.figure(figsize = plt.figaspect(0.35))
    
    ax = fig.add_subplot(111, projection = '3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,dist.reshape(N+1,N+1), cmap = 'hot')
    
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

    return dist

    
def penalisation(N, dt, ray_tub, R, Hg, Hd, Lb, angle):

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    DIST = dist(N, dt, ray_tub, R, Hg, Hd, Lb, angle)
    MASK = masque(N, ray_tub, R, Hg, Hd, Lb, angle)

    PENAL = MASK*DIST.reshape(N+1,N+1)

    np.savetxt('penal_30_5',PENAL)
    
    fig = plt.figure(figsize = plt.figaspect(0.35))
    ax = fig.add_subplot(111, projection = '3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,PENAL, cmap = 'hot')
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

    return PENAL
