import numpy as np
import scipy.sparse as sparse   # Algèbre linéaire creuse
import matplotlib.pyplot as plt # Pour les graphiques
import scipy.sparse.linalg as sci
from mpl_toolkits.mplot3d import Axes3D

def matrix(N, mu, lam):
    h = 1./N
    h2 = h*h
    taille = (1+N)*(1+N)

    

    
