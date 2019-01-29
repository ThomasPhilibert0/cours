import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def func(N):
    h = 1./N
    x1 = np.linspace(0,0+N*h,N+1)
    x2 = np.linspace(0,0+N*h,N+1)
    u = np.zeros((N+1,N+1))
    for i in np.arange(1,N-1):
        for j in np.arange(1,N-1):
            u[i][j] = x1[i]*x2[j]*((x1[i]-1)**3)*((x2[j]-1)**3)
    return u

def lap_exact(N):
    h = 1./N
    x1 = np.linspace(0,0+N*h,N+1)
    x2 = np.linspace(0,0+N*h,N+1)
    lap = np.zeros((N+1,N+1))
    for i in np.arange(1,N-1):
        for j in np.arange(1,N-1):
            lap[i][j] = 6*(1-3*x1[i]+2*(x1[i])**2)*((x2[j]-1)**3)*x2[j] + 6*(1-3*x2[j]+2*(x2[j])**2)*((x1[i]-1)**3)*x1[i]
    return lap

def laplacien(u,N):
    u_n = 0.*u
    h = 1./N
    for i in np.arange(1,N-1):
        for j in np.arange(1,N-1):
            u_n[i][j] = (u[i-1][j] - 4*u[i][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]) / (h**2)
    return u_n

def erreur1(N):
    err = np.sum((lap_exact(N) - laplacien(func(N),N))**2)/(N**2)
    return np.sqrt(err)

def erreur2(N):
    err = np.max(np.abs(lap_exact(N) - laplacien(func(N),N)))
    return err

def graph(N):
    plt.plot(lap_exact(N),'r')
    plt.plot(laplacien(func(N),N),'b')
    plt.legend()
    plt.show()
