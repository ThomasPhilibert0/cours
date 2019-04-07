import numpy as np
from matplotlib import pyplot as plt


def C(M, a):
    '''Renvoie la matrice de la fonction C appliquée à la matrice du gradient de l'image  M et de paramètre alpha = a pour le cas 2D'''
    m = np.size(M,0)
    n = np.size(M,1)    

    D = np.zeros((m,n))
    B = ngrad(M)

    for i in range (0,m-1):
        for j in range (0,n-1):
            D[i][j]  = 1./(1+(B[i][j]/a**2))

            #D[i][j] = np.exp(-(ngrad[i][j]/a)**2

    return D

def ngrad(M):

    '''Discrétisation du gradient à l'image discrète (matrice) M pour le cas 2D'''
    
    M = M.astype(float)
    m = np.size(M,0)
    n = np.size(M,1)    

    B = np.zeros((m,n))

    for i in range (0,m-1):
        for j in range (0,n-1):
            B[i][j]= (((M[i+1][j]-M[i][j]))**2)+(((M[i][j+1]-M[i][j]))**2)           
    return B
