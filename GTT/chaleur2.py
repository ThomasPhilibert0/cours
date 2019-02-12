import numpy as np
import scipy.sparse as sparse   # Algèbre linéaire creuse
import matplotlib.pyplot as plt # Pour les graphiques
import scipy.sparse.linalg as sci
from mpl_toolkits.mplot3d import Axes3D

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

def u(x1,x2):
    return x1*x2*((x1-1.)**3)*((x2-1))**3


def chaleur0(N,dt,t):

     x = np.linspace(0,1,N+1)
     y = np.linspace(0,1,N+1)

     taille1 = (N+1)*(N+1)

     T = np.zeros((t+1,taille1))          #Initialisation de la solution finale
     T[0,N+2:taille1 - N-2] = 1.
     T[0,np.arange(2*N+1, taille1, N+1)] = 0
     T[0,np.arange(2*N+2, taille1, N+1)] = 0

     for i in range (1,t):
         T[i,:] = sci.spsolve(matrix_lap2(N,dt),T[i-1,:])

     return T
     #fig = plt.figure(figsize = plt.figaspect(0.35))
    
     #ax = fig.add_subplot(1,2,1, projection = '3d')
     #X,Y = np.meshgrid(x,y)
     #ax.plot_surface(X,Y,T[1,:].reshape(N+1,N+1) ,cmap='hot')
     #plt.title("Solution au temps t = 1")
     #plt.xlabel("x")
     #plt.ylabel("y")
    
     #ax = fig.add_subplot(1,2,2, projection = '3d')
     #X,Y = np.meshgrid(x,y)
     #ax.plot_surface(X,Y,T[t-1,:].reshape(N+1,N+1),cmap='hot')
     #plt.title("Solution au temps t =" +str(t))
     #plt.xlabel("x")
     #plt.ylabel("y")

     #plt.show()


def chaleur_ex(N,dt,t):
    
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)
    
    taille1 = (N+1)*(N+1)

    V = np.zeros((t+1,taille1))                             #Allocation mémoire sol exacte
    for i in np.arange(N+1):
        for j in np.arange(N+1):
            k = i + j*(N+1)
            V[0,k] = u(x[i],y[j])

    for i in range (t):
        V[i+1,:] = sci.spsolve(matrix_lap2(N,dt),V[i,:])

    fig = plt.figure(figsize = plt.figaspect(0.35))
    
    ax = fig.add_subplot(1,2,1, projection = '3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,V[0,:].reshape(N+1,N+1) ,cmap='hot')
    plt.title("Solution au temps t = 0")
    plt.xlabel("x")
    plt.ylabel("y")
    
    ax = fig.add_subplot(1,2,2, projection = '3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,V[t,:].reshape(N+1,N+1),cmap='hot')
    plt.title("Solution au temps t =" +str(t))
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()
    
    print(np.max(V[t,:]))

def dist(N,dt):

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    T = chaleur0(N,dt,2)

    dist  = - np.log(T[2,:])*(dt*dt)

    fig = plt.figure(figsize = plt.figaspect(0.35))
    
    ax = fig.add_subplot(111, projection = '3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,dist.reshape(N+1,N+1) ,cmap='hot')
    plt.title("Solution au temps t = 0")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()
