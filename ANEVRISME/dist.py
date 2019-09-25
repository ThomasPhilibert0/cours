import numpy as np
import scipy.sparse as sparse   # Algèbre linéaire creuse
import matplotlib.pyplot as plt # Pour les graphiques
import scipy.sparse.linalg as sci
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import bmat
from math import pi
from dom import Dom_init
from dom import masque
import time

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


def dist(MAT,dt):

    """Affiche la matrice de la fonction distance de la matrice MAT (composée de 0 SAUF à l'interface (frontière) où les valeurs sont de 1) d'un domaine de taille ((N+1),(N+1))"""

    N = np.shape(MAT)[0] - 1
    
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)
    
    taille = (N+1)*(N+1)
    
    T = np.ones(taille)

    #On transforme la matrice MAT en un vecteur de taille (N+1)*(N+1)
    for i in range(1,N):
        for j in range(1,N):
            k = i*(N+1) + j
            T[k] = MAT[i][j]

    #On résoud avec 1 itération de l'équation de la chaleur
    D = sci.spsolve(matrix_lap2(N,dt), T)

    #On calcul la fonction distance
    DIST = -np.log(D)*np.sqrt(dt)

    #AFFICHAGE

    fig = plt.figure(figsize = plt.figaspect(0.5))
    
    ax = fig.add_subplot(111)
    X,Y = np.meshgrid(x,y)
    ax.contour(X,Y,DIST.reshape(N+1,N+1), cmap = 'hot')
    
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

    return DIST

    
def penalisation(MAT,MASK,dt):

    """Retourne la matrice de la fonction distance pénalisée, i.e, on affiche uniquement les 'points distances' qui sont DANS le domaine. On pénalise par notre masque du fichier dom.py.
    Penser à faire le masque en même temps que le domaine initial pour s'assure les mêmes caractères."""

    start_time = time.time()
    
    N = np.shape(MAT)[0] - 1
    
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    #On calcule la fonction distance de MAT
    DIST = dist(MAT,dt)

    #On créé la matrice de pénalisation
    PENAL = MASK*DIST.reshape(N+1,N+1)

    #Enlever le commentaire suivant pour enregistrer la matrice sous format txt.
    #np.savetxt('penal_2000_7',PENAL)

    print("Temps d'éxecution Distance_Chal : %s secondes ---" % (time.time() - start_time))
    
    #AFFICHAGE
    fig = plt.figure(figsize = plt.figaspect(0.35))
    ax = fig.add_subplot(111, projection = '3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,PENAL, cmap = 'hot')
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

    return PENAL
