import numpy as np

N = 20
h = 1./N
x1 = np.linspace(0,0+N*h,N+1)
x2 = np.linspace(0,0+N*h,N+1)

def func(x1,x2):
    u = np.zeros((N+1,N+1))
    for i in np.arange(np.size(x1)):
        for j in np.arange(np.size(x2)):
            u[i][j] = x1[i]*x2[j]*((x1[i]-1)**3)*((x2[j]-1)**3)
    return u

def lap_exact(x1,x2):
    lap = np.zeros((N+1,N+1))
    for i in np.arange(np.size(x1)):
        for j in np.arange(np.size(x2)):
            lap[i][j] = 6*(1-3*x1[i]+2*(x1[i])**2)*((x2[j]-1)**3)*x2[j] + 6*(1-3*x2[j]+2*(x2[j])**2)*((x1[i]-1)**3)*x1[i]
    return lap

def laplacien(u,h,N):
    u_n=0.*u
    for i in np.arange(1,N-1):
        for j in np.arange(1,N-1):
            u_n[i][j] = (u[i-1][j] - 4*u[i][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]) / h**2
    return u_n

def erreur(lap, u_n, N):
    e = np.abs(lap - u_n)
    err = np.max(e) 
    return np.sqrt(err)
    
