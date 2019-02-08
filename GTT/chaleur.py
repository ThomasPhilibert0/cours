import numpy as np
import scipy.sparse as sparse   # Algèbre linéaire creuse
import matplotlib.pyplot as plt # Pour les graphiques
import scipy.sparse.linalg as sci
from mpl_toolkits.mplot3d import Axes3D

def matrix_lap(N,dt):
    """Retourne une matrice qui discrétise le laplacien de u dans le domaine Omega = [xmin,xmax,ymin,ymax], découpé en N intervalles en x et y. La matrice finale est une matrice scipy.sparse CSR matrix. Cette matrice est de taille (N+1)*(N+1)"""

    h = 1./N
    h2 = h*h

    #On note les inconnues de 0 à Nx suivant x et 0 à Ny suivant y. La taille du problème est donc (Nx+1)*(Ny+1).

    #Cela correspond à x_i = i*h et y_j = j*h et la numérotation (i,j) --> k := (N+1)*j+i.

    taille = (N-1)*(N-1)

    diags = np.zeros((5,taille))

    #Diagonale principale
    diags[2,:] = -(4*dt)/h2
              
    #Diagonale "-1"
    diags[1,:] = dt/h2
    
    #Diagonale "+1"
    diags[3,:] = dt/h2

    #Diagonale "-(N+1)"
    diags[0,:] = dt/h2

    #Diagonale "+(N+1)"
    diags[4,:] = dt/h2

    #Construction de la matrice creuse
    A = sparse.spdiags(diags,[-(N-1),-1,0,1,(N-1)],taille,taille, format = "csr")

    return A

def chaleur (N,dt,t):
    
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    taille1 = (N-1)*(N-1)

    T = np.zeros((t,N+1,N+1))          #Initialisation de la solution finale
    U = np.ones(taille1)               #Initialisation matrice pour discrétisaiton
    I = sparse.eye(taille1)            #Matrice de l'identité

    LAP = (I - matrix_lap(N,dt))       #La matrice a inverser

    for i in range(t):
        U = sci.spsolve(LAP,U)
        T[i,1:N,1:N] = U.reshape(N-1,N-1).copy()

        
    fig = plt.figure(figsize = plt.figaspect(0.35))
    
    ax = fig.add_subplot(1,2,1, projection = '3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,T[1,:,:] ,cmap='hot')
    plt.title("Solution au temps t = 1")
    plt.xlabel("x")
    plt.ylabel("y")
    
    ax = fig.add_subplot(1,2,2, projection = '3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,T[t-1,:,:],cmap='hot')
    plt.title("Solution au temps t =" +str(t))
    plt.xlabel("x")
    plt.ylabel("y")

    #fig2 = plt.figure()
    #ax = fig.add_subplot(1,2,1)
    #plt.pcolormesh(x,y,T[1,:,:], cmap = 'seismic' , shading = 'flat')
    #plt.axis('image')
    #plt.draw()

    #ax = fig.add_subplot(1,2,2)
    #plt.pcolormesh(x,y,T[t-1,:,:], cmap = 'seismic', shading = 'flat')
    #plt.axis('image')
    #plt.draw()
          
    plt.show()

    
