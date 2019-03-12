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
    diags[1,:] = 0
    diags[1, N+2:taille - (N+2)] = 2./h2
    diags[1, np.arange(2*N+1, taille, N+1)] = 0
    diags[1, np.arange(2*N+2, taille, N+1)] = 0
              
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
    diags[1,:] = 0.
    diags[1, N+2:taille - (N+2)] = 2./h2
    diags[1, np.arange(2*N+1, taille, N+1)] = 0.
    diags[1, np.arange(2*N+2, taille, N+1)] = 0.

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

def resolution(N,mu,lamb,force,f_ex):
    

    #Créer des fonctions f1 et f2 pour la résolution
    #Pour une valeur de delta égale = 1 il suffit de prendre lambda =0
    
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    taille = (N+1)*(N+1)

    delta = (lamb + mu)/mu
    
    F = np.zeros(2*taille)
    E = np.zeros(2*taille)

    for i in range (N+1):
        for j in range (N+1):
            k = i + j*(N+1)
            E[k] = f_ex(x[i],y[j])[0]
            E[k + taille] = f_ex(x[i],y[j])[1]

    for i in range(1,N):
        for j in range(1,N):
            k = i + j*(N+1)
            F[k] = force(x[i],y[j],delta)[0]/mu
            F[k+taille] = force(x[i],y[j],delta)[1]/mu

    MAT = matrix_elas(N,mu,delta)

    U = sci.spsolve(MAT,F)

    u1 = E[0:taille]
    u2 = E[taille : 2*taille]
    
    U1 = U[0:taille]
    U2 = U[taille : 2*taille]
    
    fig = plt.figure(figsize = plt.figaspect(0.7))
    
    ax = fig.add_subplot(2,2,1,projection='3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y, U1.reshape((N+1,N+1)), cmap='Blues')
    plt.title("Solution de u1 discrétisée")
    plt.xlabel("x")
    plt.ylabel("y")

    ax = fig.add_subplot(2,2,2,projection='3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y, u1.reshape((N+1,N+1)), cmap='hot')
    plt.title("Solution de u1 exacte")
    plt.xlabel("x")
    plt.ylabel("y")

    
    ax = fig.add_subplot(2,2,3,projection='3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y, U2.reshape((N+1,N+1)), cmap='hot')
    plt.title("Solution de u2 discrétisée")
    plt.xlabel("x")
    plt.ylabel("y")
    
    ax = fig.add_subplot(2,2,4,projection='3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y, u2.reshape((N+1,N+1)), cmap='hot')
    plt.title("Solution de u2 exacte")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()
    
def func_exacte(x1,x2):
    u1 = x1*x2**3*(x1-1)**3*(x2-1)
    u2 = x1*x2*(x1-1)*(x2-1)**3

    return [u1,u2]

def snd_mbr(x,y,delta):
    """Les second membres F1 et F2 sont exprimés directement avec le signe - devant, ce sont les fonctions tildes, i.e pour trouver F1 et F2; F_itilde = F_i/mu. On prendra un delta egal à 1"""

    F1 = 6*(1-3*x+2*x**2)*(-1+y)*y**3 + 6*(-1+x)**3*x*y*(-1+2*y) + delta*(-1+y)*(-1+5*y-4*y**2+6*y**3+12*x**2*y**3-2*x*(-1+5*y-4*y**2+9*y**3))
    F2 = 2*(-1+y)**3*y + 6*(-1+x)*x*(1-3*y+2*y**2) + delta*(-1+x)*(y**2*(-3+4*y)+4*x**2*y**2*(-3+4*y)+x*(6-18*y+27*y**2-20*y**3))
    
    return [F1, F2]
