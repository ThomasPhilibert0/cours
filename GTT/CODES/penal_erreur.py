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

    diags[0,0:N+2] = 1
    diags[0,taille-(N+2):taille] = 1
    diags[0, np.arange(2*N+1, taille, N+1)] = 1
    diags[0, np.arange(2*N+2, taille, N+1)] = 1

    M = sparse.spdiags(diags,0,taille,taille,format = 'csr')
    
    return M

def erreur_abs(A,E,N):
    return np.max(np.abs(E - A))

def erreur_eucl(A,E,N):
    return np.sqrt(np.sum(((E-A)**2))/(N+1)**2)

def sol_ex(N):
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)
    E = np.zeros((N+1)*(N+1))
    for i in range(0,N+1):
        for j in range(0,N+1):
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
    
    for i in range (0,N+1):
        for j in range(0,N+1):
            k = i + j*(N+1)
            F[k] = -Xhi(f(x[i],y[j]),R)*func_ex(x[i],y[j])/eta + 1
            
    U = sci.spsolve(DISC,F)
    
    return U

def graphe(f,R,N,eta):
    taille = (N+1)*(N+1)
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)
    
    A = sol_penal(f,R,N,eta)
    B = sol_ex(N)

    fig = plt.figure(figsize = plt.figaspect(0.35))
    ax = fig.add_subplot(1,2,1, projection = '3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,A.reshape(N+1,N+1) ,cmap='hot')
    plt.xlabel("x")
    plt.ylabel("y")
    
    ax = fig.add_subplot(1,2,2, projection = '3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,B.reshape(N+1,N+1),cmap='hot')
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

def aff(x,b,a):
    return np.exp(b)*x**(a)

def graphe_erreur(N,eta):
    tab_err = np.zeros(N)
    tab_err2 = np.zeros(N)
    ERR1 = np.zeros(N)
    ERR2 = np.zeros(N)
    x = np.linspace(1,N,N)
    p = np.zeros((2,20))

    for a in range(1,20):
        R = a*0.05
        for i in range(1,N):        
            V = sol_penal(f, R, i+1, eta)
            U = sol_ex(i+1)
        
            tab_err[i] = erreur_abs(V,U,i)
            tab_err2[i] = erreur_eucl(V,U,i)

            x1 = x[N-10:N]
            Err1 = tab_err[N-10:N]
            Err2 = tab_err2[N-10:N]
            z1 = np.polyfit(np.log(x1),np.log(Err1),1)
            z2 = np.polyfit(np.log(x1),np.log(Err2),1)
            y1 = aff(x,z1[1],z1[0])
            y2 = aff(x,z2[1],z2[0])
            p[0,a] = z1[0]
            p[1,a] = z2[0]
            
        plt.figure(a)
        plt.plot(x,tab_err,color='blue',marker='o', linestyle='none')
        plt.plot(x,y1,color='r', linestyle='-')
        plt.plot(x,tab_err2,color='green',marker='o', linestyle='none')
        plt.plot(x,y2,color='y', linestyle='-')
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel('N')
        plt.ylabel('Erreur log Abs')
        plt.title('Pour R={R} et eta={eta}')

    plt.show()
    return p
    
