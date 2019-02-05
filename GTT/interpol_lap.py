import numpy as np
import scipy.sparse as sparse   # Algèbre linéaire creuse
import matplotlib.pyplot as plt # Pour les graphiques
import scipy.sparse.linalg as sci
import disc_lap
from mpl_toolkits.mplot3d import Axes3D


def sol_disc_inter(N):
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    F = np.zeros((N+1)*(N+1)) 

    for i in np.arange(int(N/2)): #(1,N) car on veut que ce soit 0 sur les bords
       F[i] = (i/(N+1))*((i/(N+1))-1/2)
       F[N+1-i]=((i/(N+1))-1)*((i/(N+1))-1/2)

    U = np.zeros((N+1)*(N+1))   #matrice pour la solution
    A = disc_lap.matrix_lap(N)
        
    U = sci.spsolve(A,F)
    print(U)


