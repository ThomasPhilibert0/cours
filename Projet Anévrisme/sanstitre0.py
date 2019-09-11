import numpy as np
import scipy.sparse as sparse   # Algèbre linéaire creuse
import matplotlib.pyplot as plt # Pour les graphiques
import scipy.sparse.linalg as sci
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import bmat

def Dom_init():
    
    x = np.linspace(0,1,10)
    y = np.linspace(0,1,10)   
    MAT = np.zeros((10,10))
    
    return MAT