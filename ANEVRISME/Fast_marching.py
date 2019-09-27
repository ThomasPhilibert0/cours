import skfmm
import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from dom import masque
import time

def fast_marching(MASK):

    """Retourne et affiche la matrice de la fonction distance selon la méthode du Fast_marching d'un domaine. Ce code ne fonctionne que si la matrice mise en paramètre est de type 'masque', i.e 
    une matrice avec des 0 à l'extérieur du domaine et des 1 à l'intérieur. On ajoutera un paramètre de précision pour la méthode."""

    start_time = time.time()
    
    N = np.shape(MASK)[0] - 1
    
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    MAT = np.zeros((N+1,N+1))

    #On créé la matrice relative au masque de façon à s'adapter à la fonction python
    for i in range(N+1):
        for j in range(N+1):
            if MASK[j][i] == 0 :
                MAT[j][i] = -1
            else :
                MAT[j][i] = 1

                
    #Utilisation du package skfmm 
    D = skfmm.distance(MAT, dx = 1/N)

    #On pénalise
    F = D*MASK

    #Enlever le commentaire de la ligne ci-dessous pour sauvegarder la matrice
    #np.savetxt('fast_marching_2000_2',F)

    print("Temps d'éxecution méthode Fast_Marching : %s secondes ---" % (time.time() - start_time))
    #AFFICHAGE
    
    fig = plt.figure(figsize = plt.figaspect(0.35))
    ax = fig.add_subplot(111, projection = '3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,F, cmap = 'hot')
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

    return F
