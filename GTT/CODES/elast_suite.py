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

def f_gauche(x,y,N):
    return (x-int(N/4))**2 + (y - int(N/4))**2

def f_droite(x,y,N):
    return (x-int(N/4))**2 + (y - int(3*N/4))**2

def domaine(N):
    taille = (N+1)*(N+1)

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    y_haut = int(2*N/3)
    y_bas = int(3*N/4)
    borne_gauche = int(N/4)
    borne_gaubouche = int(N/3)
    borne_droibouche = int(2*N/3)
    borne_droite = int(3*N/4)
    
    MAT = np.zeros((N+1,N+1))

    #Construction Bouche
    for i in range(y_haut,y_bas) :
        for j in range (borne_gaubouche, borne_gauche , -1):
            if i == (N - j) :
                MAT[i][j-1] = 1

    for i in range (borne_gaubouche,borne_droibouche):
        MAT[y_haut][i] = 1


    for i in range (y_haut,y_bas):
        MAT[i][i] = 1

    #Construction des yeux

    #Oeil gauche
    for i in range (N+1):
        for j in range (N+1):
            if f_gauche(i,j,N) <= int(N/12)**2 :
                MAT[i][j] = 1

    #Oeil droit
    for i in range(N+1):
        for j in range(N+1):
            if f_droite(i,j,N) <= int(N/12)**2:
                MAT[i][j] = 1
    
    return MAT


    

