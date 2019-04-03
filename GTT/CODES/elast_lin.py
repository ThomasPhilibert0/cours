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
    diags[2, N+2:taille - (N+2)] = -4./h2
    diags[2, np.arange(2*N+1, taille, N+1)] = 1.
    diags[2, np.arange(2*N+2, taille, N+1)] = 1.
              
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



def matrix_croi(N):
    """Retourne une matrice sparse de taille (N+1)*(N+1) correspondant à la discrétisation des dérivées croisées sur l'intégralité du maillage"""

    h = 1./N
    h2 = h*h
    taille = (N+1)*(N+1)

    diags = np.zeros((4,taille))

    #Diagonale "-N-2"
    diags[0, 0 : taille - 2*(N+1)] = 1./(4*h2)
    diags[0, np.arange(N-1,taille,N+1)] = 0
    diags[0, np.arange(N,taille,N+1)] = 0

    #Diagonale "-N"
    diags[1, 2 : taille - (2*N+2)] = -1./(4*h2)
    diags[1, np.arange(N+1,taille,N+1)] = 0
    diags[1, np.arange(N+2,taille,N+1)] = 0

    #Diagonale "N"
    diags[2, 2*(N+1) : taille - 2] = -1./(4*h2)
    diags[2, np.arange(2*(N+1)+(N-1),taille,N+1)] = 0
    diags[2, np.arange(2*(N+1)+N,taille,N+1)] = 0

    #Diagonale "N+2"
    diags[3, 2*(N+2) : taille] = 1./(4*h2)
    diags[3, np.arange(2*(N+2)+N-1,taille,N+1)] = 0
    diags[3, np.arange(2*(N+2)+N,taille,N+1)] = 0

    #Construction de la matrice creuse
    A = sparse.spdiags(diags,[-(N+2),-N,N,(N+2)],taille,taille, format = "csr")

    return A
    
def matrix_croi_FAUX(N):
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
    diags[1, N+2:taille - (N+2)] = -2./h2
    diags[1, np.arange(2*N+1, taille, N+1)] = 0
    diags[1, np.arange(2*N+2, taille, N+1)] = 0
              
    #Diagonale "-1"
    diags[0,N+1:taille-(N+1)] = 1./h2
    diags[0, np.arange(2*N, taille, N+1)] = 0.
    diags[0, np.arange(2*N+1, taille, N+1)] = 0.
    
    #Diagonale "+1"
    diags[2, N+3:taille-(N+1)] = 1./h2
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
    diags[1, N+2:taille - (N+2)] = -2./h2
    diags[1, np.arange(2*N+1, taille, N+1)] = 0.
    diags[1, np.arange(2*N+2, taille, N+1)] = 0.

    #Diagonale "-(N+1)"
    diags[0, 1 : taille - (2*N+3)] = 1./h2
    diags[0, np.arange(N,taille,N+1)] = 0.
    diags[0, np.arange(N+1,taille,N+1)] = 0.

    #Diagonale "+(N+1)"
    diags[2, taille - N*N + 2 : taille - 1] = 1./h2
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

def resolution(N,mu,lamb,force):
    

    #Créer des fonctions f1 et f2 pour la résolution
    #Pour une valeur de delta égale = 1 il suffit de prendre lambda =0
    
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    taille = (N+1)*(N+1)

    delta = (lamb + mu)/mu
    
    F = np.zeros(2*taille)

    for i in range(1,N):
        for j in range(1,N):
            k = i + j*(N+1)
            F[k] = force(x[i],y[j],delta)[0]/mu
            F[k+taille] = force(x[i],y[j],delta)[1]/mu

    MAT = matrix_elas(N,mu,lamb)

    U = sci.spsolve(MAT,F)
    
    return U

def solution_exacte(N,mu,lamb,f_ex):
    
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    taille = (N+1)*(N+1)

    delta = (lamb + mu)/mu
    E = np.zeros(2*taille)
    
    for i in range (N+1):
        for j in range (N+1):
            k = i + j*(N+1)
            E[k] = f_ex(x[i],y[j])[0]
            E[k + taille] = f_ex(x[i],y[j])[1]
            
    return E    
    
def graphe_reso(N,mu,lamb,force,f_ex):
    
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)
    taille = (N+1)*(N+1)
    
    U = resolution(N,mu,lamb,force)
    E = solution_exacte(N,mu,lamb,f_ex)

    u1 = E[0:taille]
    u2 = E[taille : 2*taille]
    
    U1 = U[0:taille]
    U2 = U[taille : 2*taille]
    
    
    
    fig = plt.figure(figsize = plt.figaspect(0.7))
    
    ax = fig.add_subplot(2,2,1,projection='3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y, U1.reshape((N+1,N+1)), cmap='plasma')
    plt.title("Solution de u1 discrétisée")
    plt.xlabel("x")
    plt.ylabel("y")

    ax = fig.add_subplot(2,2,2,projection='3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y, u1.reshape((N+1,N+1)), cmap='plasma')
    plt.title("Solution de u1 exacte")
    plt.xlabel("x")
    plt.ylabel("y")

    
    ax = fig.add_subplot(2,2,3,projection='3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y, U2.reshape((N+1,N+1)), cmap='plasma')
    plt.title("Solution de u2 discrétisée")
    plt.xlabel("x")
    plt.ylabel("y")
    
    ax = fig.add_subplot(2,2,4,projection='3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y, u2.reshape((N+1,N+1)), cmap='plasma')
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

def erreur_abs(A,E,N):
    return np.max(np.abs(E - A))

def erreur_eucl(A,E,N):
    return np.sqrt(np.sum((E-A)**2))/((N+1)**2)

def aff(x,b,a):
    return np.exp(b)*x**(a)

def graphe_erreur(N,mu,lamb,force,f_ex):
    tab_err1 = np.zeros(N)
    tab_err2 = np.zeros(N)
    ERR1 = np.zeros(N)
    ERR2 = np.zeros(N)
    x = np.linspace(1,N,N)

    
    for i in range(1,N+1):  
        taille = (N+1)*(N+1)
        U = resolution(N,mu,lamb,force)
        E = solution_exacte(N,mu,lamb,f_ex)
        
        E1 = E[0:taille]
        E2 = E[taille : 2*taille]
    
        U1 = U[0:taille]
        U2 = U[taille : 2*taille]
        
        tab_err1[i-1] = erreur_eucl(E1,U1,i)
        tab_err2[i-1] = erreur_eucl(E2,U2,i)

    x1 = x[N-10:N]
    Err1 = tab_err1[N-10:N]
    Err2 = tab_err2[N-10:N]
    z1 = np.polyfit(np.log(x1),np.log(Err1),1)
    z2 = np.polyfit(np.log(x1),np.log(Err2),1)
        
    y1 = aff(x,z1[1],z1[0])  
    y2 = aff(x,z2[1],z2[0])

        
        
        
        
    plt.figure(figsize = plt.figaspect(0.35))
    plt.subplot(1,2,1)
    plt.plot(x,tab_err1,color='blue',marker='o', linestyle='none')
    plt.plot(x,y1,color='r', linestyle='-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('N')
    plt.ylabel('Erreur log Abs')
    plt.title('Erreur de U1')
    plt.legend()
        
    plt.subplot(1,2,2)
    plt.plot(x,tab_err2,color='blue',marker='o', linestyle='none')
    plt.plot(x,y2,color='r', linestyle='-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('N')
    plt.ylabel('Erreur log Abs')
    plt.title('Erreur de U2')    
    plt.legend()
    
    plt.show()
    
    return z1[0],z2[0]
