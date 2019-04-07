import numpy as np
from matplotlib import pyplot as plt

def C1D(M, a):

    ''' Renvoie la matrice de la fonction C appliquée à la matrice du gradient de l'image  M et de paramètre alpha = a pour le cas 1D '''
    
    n = np.size(M,0)

    D = np.zeros(n)
    B = ngrad1D(M)

    for i in range(0,n-1):
        D[i] = 1./(1+(B[i]/a**2))

        #D[i] = np.exp(-(ngrad[i]/a)**2

    return D

def ngrad1D(M):

    ''' Discrétisation du gradient à l'image discrète (matrice) M pour le cas 1D'''
    
    M = M.astype(float)
    n = np.size(M,0)

    B = np.zeros(n)

    for i in range(0,n-1):
        B[i] = (((M[i+1]-M[i]))**2)

    return B
