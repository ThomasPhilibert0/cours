import numpy as np
import scipy.sparse as sparse   # Algèbre linéaire creuse
import matplotlib.pyplot as plt # Pour les graphiques
import scipy.sparse.linalg as sci
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import bmat

def Laplacien(N):
    """Retourne une matrice sparse de taille (N+1)*(N+1) correspondant à la discrétisation du Laplacien sur l'intégralité du maillage"""
    
    h = 1./N
    h2 = h*h
    taille = (N+1)*(N+1)

    diags = np.zeros((5,taille))

    #Diagonale principale
    diags[2,:] = 1.
    diags[2, N+2:taille - (N+2)] = 4./h2
    diags[2, np.arange(2*N+1, taille, N+1)] = 1.
    diags[2, np.arange(2*N+2, taille, N+1)] = 1.
              
    #Diagonale "-1"
    diags[1,N+1:taille-(N+1)] = -1./h2
    diags[1, np.arange(2*N, taille, N+1)] = 0.
    diags[1, np.arange(2*N+1, taille, N+1)] = 0.
    
    #Diagonale "+1"
    diags[3, N+3:taille-(N+1)] = -1./h2
    diags[3, np.arange(2*N+2, taille, N+1)] = 0.
    diags[3, np.arange(2*N+3, taille, N+1)] = 0.

    #Diagonale "-(N+1)"
    diags[0, 1 : taille - (2*N+3)] = -1./h2
    diags[0, np.arange(N,taille,N+1)] = 0.
    diags[0, np.arange(N+1,taille,N+1)] = 0.

    #Diagonale "+(N+1)"
    diags[4, taille - N*N + 2 : taille - 1] = -1./h2
    diags[4, np.arange(taille - N*N + 1 + N ,taille,N+1)] = 0.
    diags[4, np.arange(taille - N*N + 2 + N ,taille,N+1)] = 0.

    #Construction de la matrice creuse
    A = sparse.spdiags(diags,[-(N+1),-1,0,1,(N+1)],taille,taille, format = "csr")
    return A

def matrix_croi(N):
    """Retourne une matrice sparse de taille (N+1)*(N+1) correspondant à la discrétisation des dérivées croisées sur l'intégralité du maillage"""

    h = 1./N
    h2 = h*h
    taille = (N+1)*(N+1)

    diags = np.zeros((5,taille))
              
    #Diagonale "-1"
    diags[1,N+1:taille-(N+1)] = 1./h2
    diags[1, np.arange(2*N, taille, N+1)] = 0.
    diags[1, np.arange(2*N+1, taille, N+1)] = 0.
    
    #Diagonale "+1"
    diags[3, N+3:taille-(N+1)] = 1./h2
    diags[3, np.arange(2*N+2, taille, N+1)] = 0.
    diags[3, np.arange(2*N+3, taille, N+1)] = 0.

    #Diagonale "-(N+1)"
    diags[0, 1 : taille - (2*N+3)] = 1./h2
    diags[0, np.arange(N,taille,N+1)] = 0.
    diags[0, np.arange(N+1,taille,N+1)] = 0.

    #Diagonale "+(N+1)"
    diags[4, taille - N*N + 2 : taille - 1] = 1./h2
    diags[4, np.arange(taille - N*N + 1 + N ,taille,N+1)] = 0.
    diags[4, np.arange(taille - N*N + 2 + N ,taille,N+1)] = 0.

    #Construction de la matrice creuse
    A = sparse.spdiags(diags,[-(N+1),-1,0,1,(N+1)],taille,taille, format = "csr")

    return A


def der_sec1(N):
    """Retourne une matrice sparse de taille (N+1)*(N+1) correspondant à la discrétisation de la dérivée seconde par rapport à la premiere variable  sur l'intégralité du maillage"""

    # ATTENTION : Si le code ne fonctionne pas, essayer de retirer les 1 correspondant aux variables du bord.
    
    h = 1./N
    h2 = h*h
    taille = (N+1)*(N+1)

    diags = np.zeros((3,taille))

    #Diagonale principale
    #diags[1,:] = 1.
    diags[1, N+2:taille - (N+2)] = 2./h2
    #diags[1, np.arange(2*N+1, taille, N+1)] = 1.
    #diags[1, np.arange(2*N+2, taille, N+1)] = 1.
              
    #Diagonale "-1"
    diags[0,N+1:taille-(N+1)] = -1./h2
    diags[0, np.arange(2*N, taille, N+1)] = 0.
    diags[0, np.arange(2*N+1, taille, N+1)] = 0.
    
    #Diagonale "+1"
    diags[2, N+3:taille-(N+1)] = -1./h2
    diags[2, np.arange(2*N+2, taille, N+1)] = 0.
    diags[2, np.arange(2*N+3, taille, N+1)] = 0.

    #Construction de la matrice creuse
    A = sparse.spdiags(diags,[-1,0,1],taille,taille, format = "csr")

    return A

def der_sec2(N):
    """Retourne une matrice sparse de taille (N+1)*(N+1) correspondant à la discrétisation de la dérivée seconde par rapport à la seconde variable  sur l'intégralité du maillage"""
    
    h = 1./N
    h2 = h*h
    taille = (N+1)*(N+1)

    diags = np.zeros((3,taille))

    #Diagonale principale
    #diags[1,:] = 1.
    diags[1, N+2:taille - (N+2)] = 4./h2
    #diags[1, np.arange(2*N+1, taille, N+1)] = 1.
    #diags[1, np.arange(2*N+2, taille, N+1)] = 1.

    #Diagonale "-(N+1)"
    diags[0, 1 : taille - (2*N+3)] = -1./h2
    diags[0, np.arange(N,taille,N+1)] = 0.
    diags[0, np.arange(N+1,taille,N+1)] = 0.

    #Diagonale "+(N+1)"
    diags[2, taille - N*N + 2 : taille - 1] = -1./h2
    diags[2, np.arange(taille - N*N + 1 + N ,taille,N+1)] = 0.
    diags[2, np.arange(taille - N*N + 2 + N ,taille,N+1)] = 0.

    #Construction de la matrice creuse
    A = sparse.spdiags(diags,[-(N+1),0,(N+1)],taille,taille, format = "csr")

    return A
    

def matrix_elas(N,mu,lamb):
    """Retourne la matrice sparse globale pour la discrétisation du problème d'élasticité linéaire. Cette matrice sera de taille 2*(N+1)^2. Cette fonction prend en paramètre N le nbr d'intervalle de discrétisation, mu et lamb des scalaires pour l'élasticité linéaire."""

    delta = (lamb + mu)/mu

    LAP = Laplacien(N)
    CR = matrix_croi(N)
    DER1 = der_sec1(N)
    DER2 = der_sec2(N)

    MATRIX = bmat([ [LAP + delta*DER1 , CR],[-CR , LAP + delta*DER2]], format = 'csr')

    return MATRIX

def resolution(N,mu,lamb,f1,f2):

    #Créer des fonctions f1 et f2 pour la résolution
    
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    taille = (N+1)*(N+1)

    F = np.zeros(2*taille)

    for i in range (N+1):
        for j in range (N+1):
            k = i + j*(N+1)
            F[k] = -f1(x[i],y[j],lamb)/mu
            F[k+taille] = -f2(x[i],y[j],lamb)/mu

    MAT = matrix_elas(N,mu,lamb)

    U = sci.spsolve(MAT,F)

    U1 = U[0:taille]
    U2 = U[taille : 2*taille]
    
    fig1 = plt.figure(figsize = plt.figaspect(0.55))
    
    ax = fig1.add_subplot(1,2,1,projection='3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y, U1.reshape((N+1,N+1)), cmap='hot')
    plt.title("Solution de u1")
    plt.xlabel("x")
    plt.ylabel("y")


    ax = fig1.add_subplot(1,2,2,projection='3d')
    ax.plot_surface(X,Y,U2.reshape((N+1,N+1)),cmap = 'hot')
    plt.title("Solution de u2")

    plt.show()
