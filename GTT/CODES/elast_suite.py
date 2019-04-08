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
    return (x-int(3*N/4))**2 + (y - int(N/4))**2

def f_droite(x,y,N):
    return (x-int(3*N/4))**2 + (y - int(3*N/4))**2

def dom_def(N):

    ########### PENSER A METTRE UN MULTIPLE DE 12 COMME VALEUR DE N ############

    taille = (N+1)*(N+1)

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    y_haut = int(N/3)
    y_bas = int(N/4)
    borne_gauche = int(N/4)
    borne_gaubouche = int(N/3)
    borne_droibouche = int(2*N/3)
    borne_droite = int(3*N/4)
    
    MAT = np.zeros((N+1,N+1))

    #Construction Bouche pour déformation
    
    for i in range(y_bas,y_haut) :
        for j in range (borne_droibouche, borne_droite + 1):
            if i == (N - j) :
                MAT[i][j] = 1.

    for i in range (borne_gaubouche, borne_droibouche + 1):
        MAT[y_haut][i] = 1.


    for i in range (y_bas,y_haut):
        MAT[i][i] = 1.

    #fig = plt.figure(figsize = plt.figaspect(0.35))
    
    #ax = fig.add_subplot(111)
    #X,Y = np.meshgrid(x,y)
    #ax.contour(X,Y,MAT, cmap = 'magma')
    
    #plt.xlabel("x")
    #plt.ylabel("y")

    #plt.show()
    
    return MAT

def dom_init(N):

    ########### PENSER A METTRE UN MULTIPLE DE 12 COMME VALEUR DE N ############
    
    taille = (N+1)*(N+1)

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)
    
    MAT = np.zeros((N+1,N+1))

    #Construction Bouche

    for i in np.arange(int(N/4),int(3*N/4) +1):
        MAT[int(N/2)][i] = 1.

    #fig = plt.figure(figsize = plt.figaspect(0.35))
    
    #ax = fig.add_subplot(111)
    #X,Y = np.meshgrid(x,y)
    #ax.contour(X,Y,MAT, cmap = 'magma')
    
    #plt.xlabel("x")
    #plt.ylabel("y")

    #plt.show()
    
    return MAT

def BRYAN(N):

    ########### PENSER A METTRE UN MULTIPLE DE 12 COMME VALEUR DE N ############
    
    taille = (N+1)*(N+1)

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)
    
    MAT = np.zeros((N+1,N+1))

    #Construction Bouche

    for i in np.arange(int(N/4),int(3*N/4) +1):
        MAT[int(N/2)][i] = 1.

    #Construction des yeux

    #Oeil gauche
    for i in range (N+1):
        for j in range (N+1):
            if f_gauche(i,j,N) <= int(N/12)**2 :
                MAT[i][j] = 1.

    #Oeil droit
    for i in range(N+1):
        for j in range(N+1):
            if f_droite(i,j,N) <= int(N/12)**2:
                MAT[i][j] = 1.

    #fig = plt.figure(figsize = plt.figaspect(0.35))
    
    #ax = fig.add_subplot(111)
    #X,Y = np.meshgrid(x,y)
    #ax.contour(X,Y,MAT, cmap = 'magma')
    
    #plt.xlabel("x")
    #plt.ylabel("y")

    #plt.show()
    
    return MAT

def chaleurdist(MAT,N,dt,t):

     x = np.linspace(0,1,N+1)
     y = np.linspace(0,1,N+1)

     taille = (N+1)*(N+1)

     T = np.zeros((t+1,taille))          #Initialisation de la solution finale

     for i in range(N):
         for j in range(N):
             k = i + j*(N+1)
             T[:,k] = MAT[j][i]

     for i in range(t):
         T[i+1,:] = sci.spsolve(matrix_lap(N,dt), T[i,:])
         
     return T

def dist(MAT,N,dt):

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    T = chaleurdist(MAT,N,dt,1)

    dist  = -np.log(T[1,:])*np.sqrt(dt)
    dist[~np.isfinite(dist)] = 0 

    #fig = plt.figure(figsize = plt.figaspect(0.35))
    
    #ax = fig.add_subplot(111,projection = '3d')
    #X,Y = np.meshgrid(x,y)
    #ax.plot_surface(X,Y,dist.reshape(N+1,N+1), cmap = 'magma')
    
    #plt.xlabel("x")
    #plt.ylabel("y")

    #plt.show()

    return dist.reshape((N+1,N+1))


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
    """Retourne une matrice sparse de taille (N+1)*(N+1) correspondant 
    à la discrétisation des dérivées croisées sur l'intégralité du maillage"""

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


    return A

def der_sec1(N):
    """Retourne une matrice sparse de taille (N+1)*(N+1) 
    correspondant à la discrétisation de la dérivée seconde 
    par rapport à la premiere variable  sur l'intégralité du maillage"""
    
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
    """Retourne une matrice sparse de taille (N+1)*(N+1) 
    correspondant à la discrétisation de la dérivée seconde 
    par rapport à la seconde variable  sur l'intégralité du maillage"""
    
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

def second_membre(N,mu):

    taille = (N+1)*(N+1)

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)    

    DIST = dist(dom_def(N),N,0.00001)
    
    F = DIST * dom_init(N)

    S = np.zeros(2*taille)
    
    for i in range(N+1):
        for j in range(N+1):
            k = i + j*(N+1)
            S[k+ taille] = F[i][j]/mu

    return S

def resolution(N,mu,lamb):
    
    #Pour une valeur de delta égale = 1 il suffit de prendre lambda = 0
    
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    taille = (N+1)*(N+1)

    delta = (lamb + mu)/mu

    MAT = matrix_elas(N,mu,lamb)

    U = sci.spsolve(MAT,second_membre(N,mu))

    U1 = U[0:taille]
    U2 = U[taille : 2*taille]
    
    return [U1,U2]

def graphe_reso(N,mu,lamb):
    
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)
    
    taille = (N+1)*(N+1)
    
    [U1,U2] = resolution(N,mu,lamb)
        
    fig = plt.figure(figsize = [16,12])
    
    ax = fig.add_subplot(2,2,1,projection='3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y, U1.reshape(N+1,N+1), cmap='plasma')
    plt.title("Solution discrétisée U1")
    plt.xlabel("x")
    plt.ylabel("y")

    ax = fig.add_subplot(2,2,2,projection='3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y, U2.reshape(N+1,N+1), cmap='plasma')
    plt.title("Solution discrétisée")
    plt.xlabel("x")
    plt.ylabel("y")


    plt.show()

def deformation(N,mu,lamb):

    taille = (N+1)*(N+1)

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    [U1,U2] = resolution(N,mu,lamb)

    U1 = U1.reshape(N+1,N+1)
    U2 = U2.reshape(N+1,N+1)

    FINAL = np.zeros((N+1,N+1))
    Bryan = BRYAN(N)

    for i in range(N+1):
        for j in range(N+1):
            FINAL[i][j + int(np.ceil(U2[i][j]))] = Bryan[i][j]

            
    fig = plt.figure(figsize = [16,12])
    
    ax = fig.add_subplot(111)
    X,Y = np.meshgrid(x,y)
    ax.contour(X,Y, FINAL)
    plt.title("Solution discrétisée U1")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()
    
