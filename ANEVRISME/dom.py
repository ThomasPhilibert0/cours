import numpy as np
import matplotlib.pyplot as plt # Pour les graphiques
from math import pi

def Dom_init(N, ray_tub, R, Hg, Hd, Lb, angle):

    #Définition du maillage. On choisit de découper le segment [0,1] en N intervalles. Ce qui donne un problème de taille (N+1)*(N+1).
    #Prendre de préférance un N pair (pour assurer (N/2) entier)
    
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)   

    #Définition de la matrice de taille (N+1)*(N+1).
    MAT = np.zeros((N+1,N+1))
    
    
    # Construction du "sac anévrismal" centré dans le maillage et de rayon R. Ici on part du centre du maillage et on parcourt les points dans le cadran inférieur gauche.
    # On détermine les points du cercle selon la méthode suivante : si un point est dans le cercle et que celui en dessous OU à gauche est à l'extérieur alors c'est un point frontière.
    # On retire les 0 en faisant une boucle if elif
    # On complète par symétrie pour les autres cadrans.
   
    i = int(N/2)
    j = int(N/2)
    
    while (x[i]-0.5)**2 + (y[j]-0.5)**2 <= R**2:
        while (x[i]-0.5)**2 + (y[j]-0.5)**2 <= R**2:
            if j <= int(N/2) and j >= int(N/2) - ray_tub :
                MAT[i][j] = 0
                MAT[N-i][N-j] = 0
                MAT[i][N-j] = 0
                MAT[N-i][j] = 0
            elif (x[i-1]-0.5)**2 + (y[j]-0.5)**2 >= R**2 or (x[i]-0.5)**2 + (y[j-1]-0.5)**2 >= R**2:
                MAT[i][j] = 1
                MAT[N-i][N-j] = 1
                MAT[i][N-j] = 1
                MAT[N-i][j] = 1
            j = j-1
        i = i-1
        j = int (N/2)
    
    # Construction de l'artère verticale "haute" de rayon d.
    j = N
    i = int(N/2)
    
    while (x[i-ray_tub]-0.5)**2 + (y[j]-0.5)**2 >= R**2:
        MAT[j][i-ray_tub] = 1
        MAT[j][i+ray_tub] = 1
        j = j-1
    
    #Construction de l'artère horizontale gauche "haute" repérée grâce au paramètre Hg (pourcentage avec 5 en haut et 95 en bas) de rayon d.
    #On retire également les 0.
    
    y_g = 1 - (0.5-R)*Hg/100
    k = N
    
    while y[k] >= y_g :
        k = k - 1
    
    for i in range(int(N/2)-ray_tub):
        MAT[k-ray_tub][i] = 1
        MAT[k+ray_tub][i] = 1

    for r in range(k - ray_tub+1, k + ray_tub):
        MAT[r][int(N/2)-ray_tub] = 0
    

    #Construction de l'artère horizontale droite "haute" repéré grâce au paramètre Hd (compris entre 5 (tout en haut) et 95 (tout en bas)) de rayon d.
    #On parcourt en "sens inverse" i.e en partant de la droite d'où la valeur N-i.
    #On retire également les 0.
    
    y_d = 1-(0.5-R)*Hd/100
    k = N
    
    while y[k] > y_d :
        k = k - 1
    
    for i in range(int(N/2)-ray_tub):
        MAT[k-ray_tub][N-i] = 1
        MAT[k+ray_tub][N-i] = 1

    for r in range(k - ray_tub+1, k + ray_tub):
        MAT[r][int(N/2)+ray_tub] = 0

    #Construction de l'artère verticale "basse" de rayon d et de longueur Lb.
    j = 0
    i = int(N/2)
    
    while (x[i-ray_tub]-0.5)**2 + (y[j]-0.5)**2 > R**2:
        if y[j] >= 0.5 - R - Lb :
            MAT[j][i-ray_tub] = 1
            MAT[j][i+ray_tub] = 1
        j=j+1
        

    #Construction des artères obliquent d'angle "angle" (degré compris entre 5 et 85) par rapport à la verticale x = N/2

    i = int(N/2)- ray_tub
    j = 0

    théta = (angle*pi)/180
    while MAT[j][i] == 0:
        j = j + 1
        
    b = y[j] - np.tan(pi/2 - théta)*x[i]
    a = np.tan(pi/2 - théta)

    for i in range(int(N/2)-ray_tub+1):
        for k in range(j+1):
            if a*x[i] + b >= y[k]:
                if a*x[i-1] + b <= y[k] or a*x[i] + b <= y[k+1]:
                    MAT[k][i] = 1
                    MAT[k][N-i] = 1


    Y_1 = a*x[int(N/2)]+b
    Y_2 = Y_1 - ray_tub*2/(N*np.sin(théta))

    b_2 = Y_2 - a*0.5

    for i in range(int(N/2)+1):
        for k in range(j+1):
            if a*x[i] + b_2 >= y[k]:
                if a*x[i-1] + b_2 <= y[k]:
                    MAT[k][i] = 1
                    MAT[k][N-i] = 1
    
    
    #fig = plt.figure(figsize = plt.figaspect(0.35))
    #ax = fig.add_subplot(111)
    #X,Y = np.meshgrid(x,y)
    #ax.contour(X,Y,MAT, cmap = 'magma')
    #plt.xlabel("x")
    #plt.ylabel("y")
    #plt.show()
    
    return MAT
