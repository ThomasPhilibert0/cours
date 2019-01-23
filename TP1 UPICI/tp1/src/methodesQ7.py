import numpy as np
from scipy import optimize

def euler_explicite(t0,h,N, y0,f, m):
    """Méthode d'Euler pour la résolution d'un problème de Cauchy.
    Resolution de y'(t) = f(t,y(t)) avec y(t0) = y0 par la méthode d'Euler
    explicite avec un pas de temps h>0 et pour t = k*h, k=0..N.
    """

    # Construit un tableau des temps: t = t0 + [0,h,2h,...ih,...Nh]
    t = np.linspace(t0,t0+N*h,N+1) # N+1 temps de t0 à t0+N*h

    # Alloue un tableau de la taille de t, initialisé à 0. Il servira à
    # stocker la solution aux instants t0+i*h.
    y = np.zeros([m,np.size(t)])
    B = np.zeros([m])
    
    # Donnée initiale, la valeur [0] du tableau y
    y[:,0] = y0
    

    # On doit faire N itérations en temps
    for i in np.arange(m):
        for j in np.arange(N): # Boucle de 0 à N-1
            B=f(y[i,j])
            y[i,j+1] = y[i,j] + h*B[j] # Formule de la méthode d'Euler

    return B



def RK2(t0,h,N,y0,f,m):
    """Méthode de Runge-Kutta 2 pour la résolution d'un problème de Cauchy.
    Resolution de y'(t) = f(t,y(t)) avec y(t0) = y0 par la méthode de Runge-
    Kutta 2 avec un pas de temps h>0 et pour t = k*h, k=0..N.
    """

    # Construit un tableau des temps: t = t0 + [0,h,2h,...ih,...Nh]
    t = np.linspace(t0,t0+N*h,N+1) # N+1 temps de t0 à t0+N*h

    # Alloue un tableau de la taille de t, initialisé à 0. Il servira à
    # stocker la solution aux instants t0+i*h.
    y = np.zeros([m,np.size(t)])
    B = np.zeros([m])

    # Donnée initiale, la valeur [0] du tableau y
    y[:,0] = y0
    

    # On doit faire N itérations en temps
    for k in np.arange(N): # Boucle de 0 à N-1
        B=f(y)
        y[k+1] = y[k] + h*f(t[k]+0.5*h,y[k]+0.5*h*f(t[k],y[k])) # Formule de la méthode de Runge-Kutta 2

    return [t,y]

def saute_mouton(t0,h,N,y0,f):
    """Méthode de Saute-Mouton pour la résolution d'un problème de Cauchy.
    Resolution de y'(t) = f(t,y(t)) avec y(t0) = y0 par la méthode de Saute-
    Mouton avec un pas de temps h>0 et pour t = k*h, k=0..N.
    """
    
    # Construit un tableau des temps: t = t0 + [0,h,2h,...ih,...Nh]
    t = np.linspace(t0,t0+N*h,N+1) # N+1 temps de t0 à t0+N*h

    # Alloue un tableau de la taille de t, initialisé à 0. Il servira à
    # stocker la solution aux instants t0+i*h.
    y = 0.*t

    # Donnée initiale, la valeur [0] du tableau y
    y[0] = y0

    # Calcul de y_1 avec la méthode de Runge-Kutta 2
    y[1] = y[0] + h*f(t[0]+0.5*h,y[0]+0.5*h*f(t[0],y[0]))
    
    # On doit faire N itérations en temps
    for k in np.arange(1,N): # Boucle de 0 à N-1
        y[k+1] = y[k-1] + 2*h*f(t[k],y[k]) # Formule de la méthode de Saute-Mouton

    return [t,y]

def trapezePC(t0,h,N,y0,f):
    """Méthode du trapèze pour la résolution d'un problème de Cauchy.
    Resolution de y'(t) = f(t,y(t)) avec y(t0) = y0 par la méthode du
    trapèze avec un pas de temps h>0 et pour t = k*h, k=0..N.
    Première méthode de résolution: Méthode de prédiction-correction.
    """

    # Construit un tableau des temps: t = t0 + [0,h,2h,...ih,...Nh]
    t = np.linspace(t0,t0+N*h,N+1) # N+1 temps de t0 à t0+N*h

    # Alloue un tableau de la taille de t, initialisé à 0. Il servira à
    # stocker la solution aux instants t0+i*h.
    y = 0.*t

    # Donnée initiale, la valeur [0] du tableau y
    y[0] = y0
    
    # On doit faire N itérations en temps
    for k in np.arange(0,N): # Boucle de 0 à N
        A = y[k] + h*f(t[k],y[k]) # étape de prédiciton
        y[k+1] = y[k] + 0.5*h*(f(t[k],y[k]) + f(t[k+1],A) ) # étape de correction

    return [t,y]
