import numpy as np
import scipy.sparse as sparse   # Algèbre linéaire creuse
import matplotlib.pyplot as plt # Pour les graphiques
import scipy.sparse.linalg as sci
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import bmat

def matrix_lap(N,dt):
    """Utilisé pour le calcul de la distance par résolution de l'équation de la chaleur"""

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
    
def domaine(N):
    taille = (N+1)*(N+1)

    y_haut = int(2*N/3)
    y_bas = int(3*N/4)
    
    borne_gauche = int(N/4)
    borne_gaubouche = int(N/3)
    borne_droibouche = int(2*N/3)
    borne_droite = int(3*N/4)
    
    MAT = np.zeros((N+1,N+1))

    for i in range(y_haut,y_bas) :
        for j in range (borne_gaubouche, borne_gauche , -1):
            if i == (N - j) :
                MAT[i][j-1] = 1

    for i in range (borne_gaubouche,borne_droibouche):
        MAT[y_haut][i] = 1


    for i in range (y_haut,y_bas):
        MAT[i][i] = 1


    return MAT

def chaleurdist(MAT,N,dt,t):

     x = np.linspace(0,1,N+1)
     y = np.linspace(0,1,N+1)

     taille = (N+1)*(N+1)

     T = np.zeros((t+1,taille))          #Initialisation de la solution finale

     for i in range(N):
         for j in range(N):
             k = i + j*(N+1)
             T[:,k] = MAT[i][j]

     for i in range(t):
         T[i+1,:] = sci.spsolve(matrix_lap(N,dt), T[i,:])
         
     return T

def dist(MAT,N,dt):

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    T = chaleurdist(MAT,N,dt,1)

    dist  = -np.log(T[1,:])*np.sqrt(dt)
    dist[~np.isfinite(dist)] = 0 

    fig = plt.figure(figsize = plt.figaspect(0.35))
    
    ax = fig.add_subplot(111,projection = '3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,dist.reshape(N+1,N+1), cmap = 'hot')
    
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

    return dist

