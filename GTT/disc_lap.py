import numpy as np
import scipy.sparse as sparse   # Algèbre linéaire creuse
import matplotlib.pyplot as plt # Pour les graphiques
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D


def matrix_lap(N):
    """Retourne une matrice qui discrétise le laplacien de u dans le domaine Omega = [xmin,xmax,ymin,ymax], découpé en N intervalles en x et y. La matrice finale est une matrice scipy.sparse CSR matrix. Cette matrice est de taille (N+1)*(N+1)"""

    h = 1./N
    h2 = h*h

    #On note les inconnues de 0 à Nx suivant x et 0 à Ny suivant y. La taille du problème est donc (Nx+1)*(Ny+1).

    #Cela correspond à x_i = i*h et y_j = j*h et la numérotation (i,j) --> k := (N+1)*j+i.

    taille = (1+N)*(1+N)

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
    
def f(x1,x2):
    return 6.*(1.-3.*x1+2.*x1**2)*((x2-1.)**3)*x2 + 6.*(1.-3.*x2+2.*x2**2)*((x1-1.)**3)*x1

def u(x1,x2):
    return x1*x2*((x1-1.)**3)*((x2-1))**3

def sol_disc(N):
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    F = np.zeros((N+1)*(N+1))   #Allocation mémoire de f
    V = np.zeros((N+1)*(N+1))   #Allocation mémoire sol exacte

    for i in np.arange(1,N):
        for j in np.arange(1,N):
            k = i + j*(N+1)
            F[k] = f(x[i],y[j])

    for i in np.arange(N+1):
        for j in np.arange(N+1):
            k = i + j*(N+1)
            V[k] = u(x[i],y[j])

    U = np.zeros((N+1)*(N+1)) #matrice pour la solution
    A = matrix_lap(N)

    #plt.spy(A)
    #plt.show()
        
    U = spsolve(A,F)

    fig = plt.figure(1)
    ax = Axes3D(fig)
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y, U.reshape((N+1,N+1)), cmap='hot')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Solution discrétisée")
    
    fig2 = plt.figure(2)
    ax = Axes3D(fig2)
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,V.reshape((N+1,N+1)),cmap = 'hot')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Solution exacte")
    
    plt.show()
    
    err2 = np.sum((V-U)**2)/((N+1)**2)
    err1 = np.max(np.abs(V - U))    

    print("{:8s} {:12s}".
          format("Taille", "erreur demandée"))
    print("{:8d} {:12.5e}".
          format((N+1)*(N+1), err2))
    
    print("{:8s} {:12s}".
          format("Taille", "erreur absolue"))
    print("{:8d} {:12.5e}".
          format((N+1)*(N+1), err1))
    
