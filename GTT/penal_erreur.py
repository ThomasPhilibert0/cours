import numpy as np
import scipy.sparse as sparse   # Algèbre linéaire creuse
import matplotlib.pyplot as plt # Pour les graphiques
import scipy.sparse.linalg as sci

def matrix_lap(N):
    """Retourne une matrice qui discrétise le laplacien de u dans le domaine Omega = [xmin,xmax,ymin,ymax], découpé en N intervalles en x et y. La matrice finale est une matrice scipy.sparse CSR matrix. Cette matrice est de taille (N+1)*(N+1)"""

    h = 1./N
    h2 = h*h

    #On note les inconnues de 0 à Nx suivant x et 0 à Ny suivant y. La taille du problème est donc (Nx+1)*(Ny+1).

    #Cela correspond à x_i = i*h et y_j = j*h et la numérotation (i,j) --> k := (N+1)*j+i.

    taille = (1+N)*(1+N)

    diags = np.zeros((5,taille))

    #Diagonale principale
    diags[2,:] = 1.
    diags[2, N+2:taille - (N+2)] = -4./h2 
    diags[2, np.arange(2*N+1, taille, N+1)] = 1.
    diags[2, np.arange(2*N+2, taille, N+1)] = 1.
              
    #Diagonale "-1"
    diags[1,N+1:taille-(N+1)] = 1./h2
    diags[1, np.arange(2*N, taille, N+1)] = 0.
    diags[1, np.arange(2*N+1, taille, N+1)] = 0.
    
    #Diagonale "+1"
    diags[3, N+3:taille-(N+1)] = 1./h2
    diags[3, np.arange(2*N+2, taille, N+1)] = 0.
    diags[3, np.arange(2*N+3, taille, N+1)] = 0.

    #Diagonale "-(N+1)"
    diags[0, 1 : taille - (2*N+3)] = 1./h2
    diags[0, np.arange(N,taille,N+1)] = 0.
    diags[0, np.arange(N+1,taille,N+1)] = 0.

    #Diagonale "+(N+1)"
    diags[4, taille - N*N + 2 : taille - 1] = 1./h2
    diags[4, np.arange(taille - N*N + 1 + N ,taille,N+1)] = 0.
    diags[4, np.arange(taille - N*N + 2 + N ,taille,N+1)] = 0.

    #Construction de la matrice creuse
    A = sparse.spdiags(diags,[-(N+1),-1,0,1,(N+1)],taille,taille, format = "csr")

    return A

def func_ex(x,y):
    return 0.25*((x-0.5)**2 + (y-0.5)**2)

def f(x,y):
    return (x-0.5)**2 + (y-0.5)**2

def Xhi(f,R):
    """R est le rayon du masque"""
    if f <= R**2:
        return 0
    else:
        return 1
    
def masque(f,R,N):
    """Retourne une matrice qui discrétise le masque du domaine grâce à la fonction Xhi précédente, la matrice créée est une matrice sparse de taille (N+1)*(N+1)"""

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)
    
    taille = (N+1)*(N+1)
    
    diags = np.zeros((1,taille))
    
    #Diagonale principale
    for i in range(N+1):
        for j in range(N+1):
            k = i + j*(N+1)
            diags[0,k] = Xhi(f(x[i],y[j]),R)

    diags[0,0:N+2] = 0
    diags[0,taille-(N+2):taille] = 0
    diags[0, np.arange(2*N+1, taille, N+1)] = 0
    diags[0, np.arange(2*N+2, taille, N+1)] = 0

    M = sparse.spdiags(diags,0,taille,taille,format = 'csr')
    
    return M

def erreur_abs(A,E,N):
    return np.max(np.abs(E - A))

def sol_ex(N):
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)
    E = np.zeros((N+1)*(N+1))
    for i in range(1,N):
        for j in range(1,N):
            k = i + j*(N+1)
            E[k] = func_ex(x[i],y[j])
    return E

def sol_penal(f,R,N,eta):
    taille = (N+1)*(N+1)
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)    

    A = matrix_lap(N)
    B = masque(f,R,N)

    DISC = (A - (1/eta)*B)
        
    F = np.zeros(taille)
    
    for i in range (1,N):
        for j in range(1,N):
            k = i + j*(N+1)
            F[k] = -Xhi(f(x[i],y[j]),R)*func_ex(x[i],y[j])/eta + 1
            
    U = sci.spsolve(DISC,F)
    
    return U

def graphe_erreur(N,R,eta):
    tab_err = np.zeros(N)

    for i in range(1,N):        
        V = sol_penal(f, R, i+1, eta)
        U = sol_ex(i+1)
        
        tab_err[i] = erreur_abs(V,U,i)

        plt.plot(i,tab_err[i],color='blue',marker='o')
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel('N')
        plt.ylabel('Erreur log Abs')
        plt.title(f'Pour R={R} et eta={eta}')

    plt.show()