import numpy as np
import scipy.sparse as sparse 
import numpy.linalg as alg
import math
from matplotlib import pyplot as plt


#Réflexion des bords pour le cas 1D
def bord1D(U, dt):

    '''Effectue les conditions de Neumann sur les bords de l'image U et un pas de temps dt '''

    U_n = np.copy(U)
    N = np.size(U_n)
    U_n[0] = dt * (U[1] - U[0]) + U[0]
    U_n[N-1] = dt * (U[N-2] - U[N-1]) + U[N-1]

    return U_n
    
    
#Réflexion des bords pour le cas 2D
def bord2D(U,d):

    ''' Renvoie une image étendue par reflexion sur les bords. Le parametre d est le nombre de pixel ajouté sur chaque bord. L'image finale est donc de taille +2d dans chaque direction. '''

    m = np.size(U,0)
    n = np.size(U,1)
    
    # m et n sont dimensions de l'image U

    M = m + 2*d
    N = n + 2*d
    
    #M et N sont les dimensions de l'image avec réflexion

    B = np.zeros((M,N))
    B[d:M-d, d:N-d] = np.copy(U)

    #On vient de compléter B avec l'image U
    #Maintenant on complète la réflexion

    for i in range(1, m):
       for j in range(1, d):
          B[i+d][j] = U[i][d-j+1]

       for j in range(N-d+1, N):
          B[i+d][j] = U[i][n+N-j-d]

    for j in range(1, N):
        for i in range(1, d):
            B[i][j] = B[2*d-i+1][j]
        for i in range(M-d+1, M):
            B[i][j] = B[2*M-i-2*d][j]
    
    return B



