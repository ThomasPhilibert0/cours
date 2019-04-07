import numpy as np
from matplotlib import pyplot as plt


#Une valeur de mu = 100 et sigma = 20.

def bruit (u, mu, sigma):

    '''Renvoie l'image discrète bruitée par une fonction gaussienne de moyenne mu et d'écart-type racine de sigma'''
    
    #u =u[:,:,0](si image divisée ++ arrays)

    v=sigma*np.random.randn(np.size(u,0),np.size(u,1))+mu
    b=u+v

    return b
