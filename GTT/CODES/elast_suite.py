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

def f(x,y):
    return y - 0.5*(x**2)

def gamma(f):
    """Fonction qui indique si on est sur la courbe gamma d'équation f"""
    if f == 0:
        return 1
    else:
        return 0
    
def domaine(f,N):
    """Fonction qui discretise le domaine carré avec gamma dedans (ca vaut 0 partout dans le carré et 1 sur la courbe gamma"""
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    taille = (N+1)*(N+1)

    diags = np.zeros((1,taille))

    for i in range(N-2,N+1):
        for j in range(N-2,N+1):
            k = i + j*(N+1)
            diags[0,k] = gamma(f(x[i],y[j]))

    D = sparse.spdiags(diags,0,taille,taille,format = 'csr')
    return D

    
def dist_gamma(f,N):
    """Retourne la fonction distance de Gamma avec T=0 sur le bord de domaine"""

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    taille = (N+1)*(N+1)

    T = np.zeros((2,taille))

    
    
    

