import numpy as np
from scipy import optimize

def euler_explicite(t0,h,N, y0,f, m, A):
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
    C = np.zeros([m])
    # Donnée initiale, la valeur [0] du tableau y
    y[:,0] = y0
    

    # On doit faire N itérations en temps
    for j in np.arange(N):# Boucle de 0 à N-1
        C=y[:,j]
        B=f(t,A,C)
        for i in np.arange(m):            
            
            y[i,j+1] = y[i,j] + h*B[i] # Formule de la méthode d'Euler

    return [t,y]






def RK2(t0,h,N,y0,f,m, A):
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
    C = np.zeros([m])
    D = np.zeros([m])

    # Donnée initiale, la valeur [0] du tableau y
    y[:,0] = y0
    

    # On doit faire N itérations en temps
    for j in np.arange(N): # Boucle de 0 à N-1
        C=y[:,j]
        B=C + 0.5*h*f(t[j],A,C)
        D = f(t[j]+0.5*h,A,B)
        for i in np.arange(m):
            y[i,j+1] = y[i,j] + h*D[i] # Formule de la méthode de Runge-Kutta 2

    return [t,y]

def saute_mouton(t0,h,N,y0,f,m,A):
    """Méthode de Saute-Mouton pour la résolution d'un problème de Cauchy.
    Resolution de y'(t) = f(t,y(t)) avec y(t0) = y0 par la méthode de Saute-
    Mouton avec un pas de temps h>0 et pour t = k*h, k=0..N.
    """
    
    # Construit un tableau des temps: t = t0 + [0,h,2h,...ih,...Nh]
    t = np.linspace(t0,t0+N*h,N+1) # N+1 temps de t0 à t0+N*h

    # Alloue un tableau de la taille de t, initialisé à 0. Il servira à
    # stocker la solution aux instants t0+i*h.
    y = np.zeros([m,np.size(t)])
    B = np.zeros([m])
    C = np.zeros([m])
    D = np.zeros([m])

    # Donnée initiale, la valeur [0] du tableau y
    y[:,0] = y0

    # Calcul de y_1 avec la méthode de Runge-Kutta 2
    C=y[:,0]
    D=f(t[0]+0.5*h,A,C+0.5*h*f(t[0],A,C))
    for i in np.arange(m):
        y[i,1] = y0[i] + h*D[i]
    
    # On doit faire N itérations en temps
    for j in np.arange(1,N): # Boucle de 0 à N-1
        C=y[:,j]
        B=f(t[j],A,C)
        for i in np.arange(m):            
            y[i,j+1] = y[i,j-1] + 2*h*B[i] # Formule de la méthode de Saute-Mouton

    return [t,y]

def trapezePC(t0,h,N,y0,f,m,A):
    """Méthode du trapèze pour la résolution d'un problème de Cauchy.
    Resolution de y'(t) = f(t,y(t)) avec y(t0) = y0 par la méthode du
    trapèze avec un pas de temps h>0 et pour t = k*h, k=0..N.
    Première méthode de résolution: Méthode de prédiction-correction.
    """

    # Construit un tableau des temps: t = t0 + [0,h,2h,...ih,...Nh]
    t = np.linspace(t0,t0+N*h,N+1) # N+1 temps de t0 à t0+N*h

    # Alloue un tableau de la taille de t, initialisé à 0. Il servira à
    # stocker la solution aux instants t0+i*h.
    y = np.zeros([m,np.size(t)])
    C = np.zeros([m])
    D = np.zeros([m])
    E = np.zeros([m])
    F = np.zeros([m])
    # Donnée initiale, la valeur [0] du tableau y
    y[:,0] = y0
    
    # On doit faire N itérations en temps
    for j in np.arange(N): # Boucle de 0 à N
        C=y[:,j]
        D = C + h*f(t[j],A,C)# étape de prédiciton        
        E = f(t[j],A,C)
        F = f(t[j+1],A,D)
        for i in np.arange(m):
            y[i,j+1] = y[i][j] + 0.5*h*(E[i] + F[i] ) # étape de correction

    return [t,y]
