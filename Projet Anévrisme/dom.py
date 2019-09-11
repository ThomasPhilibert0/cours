import numpy as np
import scipy.sparse as sparse   # Algèbre linéaire creuse
import matplotlib.pyplot as plt # Pour les graphiques
import scipy.sparse.linalg as sci
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import bmat

def Dom_init(N,d, R, Hg, Hd, Lb, théta):
    
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)   
    MAT = np.zeros((N,N))
    
    
    # Construction du "sac anévrismal" centré dans le maillage et de rayon R
    i = int(N/2)
    j = int(N/2)
    
    while (x[i]-0.5)**2 + (y[j]-0.5)**2 <= R**2:
        while (x[i]-0.5)**2 + (y[j]-0.5)**2 <= R**2:
            if (x[i-1]-0.5)**2 + (y[j]-0.5)**2 >= R**2 or (x[i]-0.5)**2 + (y[j-1]-0.5)**2 >= R**2:
                MAT[i][j] = 1
                MAT[N-i][N-j] = 1
                MAT[i][N-j] = 1
                MAT[N-i][j] = 1
            j = j-1
        i = i-1
        j = int(N/2)
        
    # Construction de l'artère horizontale "haute" de diamètre d
    j = N-1
    i = int(N/2)
    
    while (x[i-2]-0.5)**2 + (y[j]-0.5)**2 >= R**2:
        MAT[j][i-d] = 1
        MAT[j][i+d] = 1
        j=j-1
        
    #Construction de l'artère verticale gauche "haute" repéré grâce au paramètre Hg (compris entre 0 (tout en haut) et 100 (tout en bas)) de diamètre d
    y_g = 1-(0.5-R)*Hg/100
    k = N-1
    
    while y[k] > y_g :
        k = k - 1
    
    for i in range(int(N/2)-d):
        MAT[k-d][i] = 1
        MAT[k+d][i] = 1
        
    #Construction de l'artère verticale droite "haute" repéré grâce au paramètre Hd (compris entre 0 (tout en haut) et 100 (tout en bas)) de diamètre d
    y_d = 1-(0.5-R)*Hd/100
    k = N-1
    
    while y[k] > y_d :
        k = k - 1
    
    for i in range(int(N/2)-d):
        MAT[k-d][N-1-i] = 1
        MAT[k+d][N-1-i] = 1
    
    #Construction de l'artère verticale "basse" de diamètre d et de longueur Lb
    j = 0
    i = int(N/2)
    
    while (x[i-2]-0.5)**2 + (y[j]-0.5)**2 >= R**2:
        if y[j] >= 0.5 - R - Lb :
            MAT[j][i-d] = 1
            MAT[j][i+d] = 1
        j=j+1
        
    #Construction des artères obliquent d'angle Théta par rapport à la verticale
    
    
    
    
    fig = plt.figure(figsize = plt.figaspect(0.35))
    ax = fig.add_subplot(111)
    X,Y = np.meshgrid(x,y)
    ax.contour(X,Y,MAT, cmap = 'magma')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    return MAT