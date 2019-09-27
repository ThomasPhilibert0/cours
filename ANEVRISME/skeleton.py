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
import time


def skeleton_maxloc(DIST,MASK,lim) :

    """Affiche le skeleton selon une méthode de maximum local.En paramètre on prend la matrice de la fonction distance et lim variant de 0 à 12.
    Ici on sélectionne 12 directions et la valeur du point du maillage oscille entre 0 et 12 selon dans combien de direction il est un maximum local."""

    #METHODE FIABLE A NE PLUS TOUCHER

    start_time = time.time()
    
    N = np.shape(DIST)[0] - 1
    
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)
    epsilon = 10**(-16)
    DIR = 0

    SKELET = np.zeros((N+1,N+1))
    
    for i in range(2,N-1):
        for j in range(2,N-1):
            pix = DIST[i][j]


            for k in range(0,3):
                for l in range(0,3):
                    if k == 0 and l == 0:
                        DIR = DIR
                    else :
                        if k == 0 or l == 0:
                            if pix > DIST[i+k][j+l] - epsilon and pix > DIST[i-k][j-l] - epsilon:
                                DIR += 1
                        else:
                            if pix > DIST[i+k][j+l] - epsilon and pix > DIST[i-k][j-l] - epsilon:
                                DIR += 1
                            if pix > DIST[i+k][j-l] - epsilon and pix > DIST[i-k][j+l] - epsilon:
                                DIR += 1

            if DIR >= lim :
                SKELET[i][j] = DIR

            DIR = 0

    B = SKELET*MASK

    print("Temps d'éxecution méthode Max_Loc : %s secondes ---" % (time.time() - start_time))
    
    fig1 = plt.figure(figsize = plt.figaspect(0.5))
    ax = fig1.add_subplot(111)
    X,Y = np.meshgrid(x,y)
    ax.contourf(X,Y,B, cmap = 'binary')
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
    plt.title("Gradient de la fonction distance")
 

    plt.show()

def skeleton_grad(DIST,MASK,lim):
    """Retourne le skeleton selon la méthode de maximum local via le gradient. Si il y a un changement de signe pour a dérivée en x ou en y alors c'est un max local. On ajoute +1 à chaque fois que
    c'est un max local dans une des 2 dérivées."""
    
    N = np.shape(DIST)[0] - 1
    epsilon = 10**(-10)
    DIR = 0
    
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    #Calcul du gradient
    dy,dx = np.gradient(DIST)
    
    SKELET = np.zeros((N+1,N+1))

    #Les conditions de max local
    for i in range(1,N):
        for j in range(1,N):
            if dx[i][j-1]*dx[i][j+1] - epsilon < 0:
                DIR += 1 
            if dy[i-1][j]*dy[i+1][j] - epsilon < 0:
                DIR += 1 
            if dx[i-1][j-1]*dx[i+1][j+1] < epsilon or dy[i-1][j-1]*dy[i+1][j+1] < epsilon :  
                DIR += 1
            if dx[i-1][j+1]*dx[i+1][j-1] < epsilon or dy[i-1][j+1]*dy[i+1][j-1] < epsilon :  
                DIR += 1

            if DIR >= lim :
                SKELET[i][j] = DIR

            DIR = 0

    SKELET = SKELET*MASK
    
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



def isolation(SKELET,K):

    start_time = time.time()

    N = np.shape(SKELET)[0] - 1

    diam = K

    ISO = np.zeros((N+1,N+1))
    
    ligne,colonne = np.nonzero(SKELET)
    
    for i in range(np.size(ligne)):
        l = ligne[i]
        c = colonne[i]
        cpt = 0
        
        pix = SKELET[l][c]
        if l > diam + 1 and l < N - diam and c > diam + 1 and c < N - diam:
            k = -diam
            while k <= diam and cpt <= 1 :
                m = -diam
                while m <= diam and cpt <= 1:
                    if SKELET[l+k][c+m] != 0 :
                        cpt += 1
                    m += 1
                k += 1
            if cpt <= 1:
                ISO[l][c] = 1

    print("Temps d'éxecution méthode d'isolation : %s secondes ---" % (time.time() - start_time))

    
            
    #AFFICHAGE
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)
    
    fig = plt.figure(figsize = plt.figaspect(0.35))
    ax = fig.add_subplot(111)
    X,Y = np.meshgrid(x,y)
    ax.contourf(X,Y,ISO, cmap = 'binary')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Points d'intersection")

    plt.show()

    return np.nonzero(ISO)


def skeleton_maxloc_opti(DIST):

    start_time = time.time()
    
    N = np.shape(DIST)[0] - 1

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    ligne,colonne = np.nonzero(DIST)

    SKELET = np.zeros((N+1,N+1))

    epsilon = 10**(-16)

    for i in range(np.size(ligne)):

        l = ligne[i]
        c = colonne[i]

        pix = DIST[l][c]

        cpt = 0
        
        if l > 3 and l < N - 3 and c > 3 and c < N - 3 :

            k = -2

            while k <= 2 :

                m = -2

                while m <= 2 and pix > DIST[l+k][c+m] - epsilon : 

                    if m == 0 and k == 0 :

                        cpt = cpt

                    else:

                        cpt += 1

                    m += 1

                k += 1

        
            if cpt >= 24 :

                SKELET[l][c] = 1

        
    print("Temps d'éxecution méthode d'isolation : %s secondes ---" % (time.time() - start_time))
    
    fig = plt.figure(figsize = plt.figaspect(0.35))
    ax = fig.add_subplot(111)
    X,Y = np.meshgrid(x,y)
    ax.contourf(X,Y,SKELET, cmap = 'binary')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Points d'intersection")

    plt.show()

    return SKELET
