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


def skeleton_maxloc(DIST,lim) :

    """Affiche le skeleton selon une méthode de maximum local.En paramètre on prend la matrice de la fonction distance et lim variant de 0 à 4.
    Ici on sélectionne 4 directions et la valeur du point du maillage oscille entre 0 et 4 selon dans combien de directionil est un maximum local."""

    N = np.shape(DIST)[0] - 1
    
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    DIR = 0
    dir1 = 0
    dir2 = 0
    dir2 = 0
    dir3 = 0
    dir4 = 0

    SKELET = np.zeros((N+1,N+1))
    
    for i in range(1,N):
        for j in range(1,N):
            pix = DIST[i][j]

            #Première direction(verticale)
            if pix > DIST[i-1][j] and pix > DIST[i+1][j]:
                dir1 = 1
            #Seconde direction(horizontale)
            if pix > DIST[i][j-1] and pix > DIST[i][j+1]:
                dir2 = 1
            #Troisieme direction(diagonale g_d)
            if pix > DIST[i-1][j-1] and pix > DIST[i+1][j+1]:
                dir3 = 1
            #Quatrième direction(diagonale d_g)
            if pix > DIST[i+1][j-1] and pix > DIST[i-1][j+1]:
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

    fig1 = plt.figure(figsize = plt.figaspect(0.5))
    ax = fig1.add_subplot(111)
    X,Y = np.meshgrid(x,y)
    ax.contourf(X,Y,SKELET, cmap = 'binary')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Skeleton par méthode du Max Local")

    plt.show()

    return SKELET

def aff_grad(DIST) :

    """Affiche la représentation du gradient pour la matrice de la fonction distance donnée en paramètre """
    N = np.shape(DIST)[0] - 1
    
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    #Calcul du gradient
    dy,dx = np.gradient(DIST)

    #AFFICHAGE
    fig = plt.figure(figsize = plt.figaspect(0.35))
    ax = fig.add_subplot(111)
    X,Y = np.meshgrid(x,y)
    ax.quiver(X,Y,dx,dy)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gradient de la fonciton distance")
 

    plt.show()

def skeleton_grad(DIST):
    """Retourne le skeleton selon la méthode de maximum local via le gradient. Si il y a un changement de signe pour a dérivée en x ou en y alors c'est un max local. On ajoute +1 à chaque fois que
    c'est un max local dans une des 2 dérivées."""
    
    N = np.shape(DIST)[0] - 1

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    #Calcul du gradient
    dy,dx = np.gradient(DIST)
    
    SKELET = np.zeros((N+1,N+1))

    #Les conditions de max local
    for i in range(1,N):
        for j in range(1,N):
            if dx[j][i-1]*dx[j][i+1] < 0:
                SKELET[j][i] = SKELET[j][i]+1
            if dy[j-1][i]*dy[j+1][i] < 0:
                SKELET[j][i] = SKELET[j][i]+1

    #AFFICHAGE
    fig = plt.figure(figsize = plt.figaspect(0.35))
    ax = fig.add_subplot(111)
    X,Y = np.meshgrid(x,y)
    ax.contourf(X,Y,SKELET, cmap = 'binary')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Skeleton par méthode du gradient")

    plt.show()

    return SKELET

def combinaison(DIST, precision):
    """Construit la matrice du skeleton par combinaison des 2 méthodes précédentes. On a donc une précision variant entre 1 et 6."""

    N = np.shape(DIST)[0] - 1
    
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    MAT_1 = skeleton_grad(DIST)
    MAT_2 = skeleton_maxloc(DIST,1)

    MAT = MAT_1 + MAT_2
    
    for i in range(N+1):
        for j in range(N+1):
            if MAT[i][j] < precision :
                MAT[i][j] = 0

    #AFFICHAGE
    fig = plt.figure(figsize = plt.figaspect(0.35))
    ax = fig.add_subplot(111)
    X,Y = np.meshgrid(x,y)
    ax.contourf(X,Y,MAT, cmap = 'binary')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Skeleton par combinaison des 2 méthodes")

    plt.show()
