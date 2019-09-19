import skfmm
import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from dom import masque


def distance(N, ray_tub, R, Hg, Hd, Lb, angle):

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)
    MAT = np.zeros((N+1,N+1))
    

    MASK = masque(N, ray_tub, R, Hg, Hd, Lb, angle)
    for i in range(N+1):
        for j in range(N+1):
            if MASK[j][i] == 0 :
                MAT[j][i] = -1
            else :
                MAT[j][i] = 1
                

    D = skfmm.distance(MAT, dx=1e-7)
    F = D*MASK



    fig = plt.figure(figsize = plt.figaspect(0.35))
    ax = fig.add_subplot(111, projection = '3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,F, cmap = 'hot')
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()
    return MAT
