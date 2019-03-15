import numpy as np
import matplotlib.pyplot as plt # Pour les graphiques
import scipy.sparse.linalg as sci
import disc_lap


alpha = -1.
beta = -1.

def fb(x,y):
    return x*(x-(1./2.))*alpha

def fh(x,y):
    return (x-1.)*(x-(1./2.))*beta

def sol_disc_inter(N):
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)
    F = np.zeros((N+1)*(N+1)) 

    for i in np.arange(0,N+1):    #(1,N) car on veut que ce soit 0 sur les bords
        for j in np.arange(0,N+1):
            k = i + j*(N+1)
            if (i <= (N/2.) and j == 0):
                F[k] = fb(x[i],y[j])
            elif (i >= (N/2.) and j == N):
                F[k] = fh(x[i],y[j])
                

    U = np.zeros((N+1)*(N+1))   #matrice pour la solution
    A = disc_lap.matrix_lap(N)
        
    U = sci.spsolve(A,F)
    return U

def affichage_inter(N):
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)
    
    U = sol_disc_inter(N)

    fig = plt.figure(figsize = plt.figaspect(0.35))
    
    ax = fig.add_subplot(1,2,1,projection='3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y, U.reshape((N+1,N+1)), cmap='hot')
    plt.title("Solution discrétisée")
    plt.xlabel("x")
    plt.ylabel("y")

