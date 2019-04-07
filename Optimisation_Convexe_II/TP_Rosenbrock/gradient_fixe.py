import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def f(x):
    return (x[0] - 1)**2 + 10*(x[0]**2 - x[1])**2

def grad_f(x):
    return np.array([2*(x[0] - 1)+40*x[0]*(x[0]**2 - x[1]), -20*(x[0]**2 - x[1])])

def Hess_f(x):
    return np.array([[2 + 40*(x[0]**2 - x[1]) + 80*x[0]**2, -40*x[0]],[-40*x[0], 20]])

def f_1d(x, d, t):
    return f(x+d*t)

def df_1d(x, d, t):
    return np.dot(grad_f(x+t*d),d)

def d2f_1d(x, d, t):
    return np.dot(np.dot(Hess_f(x+d*t),d),d)

def Newton_1D(x, d, t0, eps, n_max):
    
    n_iter = 0
    if d2f_1d(x,d,t0) == 0 :
        print('impossible')
    else:
        der2 = d2f_1d(x,d,t0)
        der1 = df_1d(x, d, t0)
        t = t0 - der1/der2
        
        while np.linalg.norm(t0-t) > eps and n_iter < n_max:
            
            t0 = t
            der2 = d2f_1d(x, d, t0)
            der1 = df_1d(x, d, t0)
            if der2 == 0:
                break
            else:
                t = t0 - der1/der2
                if f_1d(x, d, t)>f_1d(x, d, t0):
                    t = t0 + der1/der2
            n_iter = n_iter + 1
        
        return t
    
def gradient(x, Rho):
    return x - np.dot(Rho, grad_f(x))

def gradient_fixe(x0, Rho, eps, n_max):
    
    n_iter = 0
    x = gradient(x0, Rho)
    
    while np.linalg.norm(x0-x) >= eps and n_iter < n_max :
        
        x0 = x
        x = gradient(x0, Rho)
        n_iter = n_iter + 1
    
    return [x,n_iter]

def gradient_opti(x0, eps, n_max):
    
    n_iter = 0
    G_f = grad_f(x0)
    Rho = np.abs(Newton_1D(x0, G_f,1, 0.00000001, 150))
    x = gradient(x0, Rho)
    
    while np.linalg.norm(x0-x) >= eps and n_iter < n_max :
        
        x0 = x
        G_f = grad_f(x0)
        Rho = np.abs(Newton_1D(x0, G_f,1, 0.00000001, 150))
        x = gradient(x0, Rho)
        n_iter = n_iter + 1
        
    return x

def Newton_2D(x0, eps, n_max):
    
    n_iter = 0
    Rho = np.linalg.inv(Hess_f(x0))
    x = gradient(x0, Rho)
    
    while np.linalg.norm(x0-x) >= eps and n_iter < n_max :
        
        x0 = x
        Rho = np.linalg.inv(Hess_f(x0))
        x = gradient(x0, Rho)
        n_iter = n_iter + 1
        
    return [x,n_iter]
    
    
    
    
    
    
    
