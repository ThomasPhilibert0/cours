import numpy as np
import matplotlib.pyplot as plt

import methodes
import equations

# Résolution approchée de y' = 1-y^2 avec y(0) = 0 pour t dans [0,1]

equations.a = -1.
equations.b = 1.
t0,y0 = 0.,0.
T = 1.
h = 0.2

print("Valeur de l'erreur en fonction de h")

#Définition de la boucle for
for i in range(1,6,1) :
    N = int(T/h) #Nombre d'itérations

    [t,y1] = methodes.trapezePC(t0,h,N,y0,equations.f_poly)

    # Solution exacte aux mêmes instants
    z1 = equations.sol_poly1(t)

    # Calcul de l'erreur maximum relative
    e1 = np.nanmax(np.abs((z1-y1)/z1))

    #Graphe des solutions exactes et approchées
    plt.figure(1)
    plt.plot(t,y1,'b-+')
    plt.plot(t,z1,'r')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title("Méthode du trapeze PC")

    # Écriture de l'erreur en fonction de h
    print("{0} | {1}".format(h,e1))
    plt.figure(2)
    plt.plot(h,e1,'b-+',label = 'Pour h = '+ str(h))

    h = h/2

plt.xscale('log')
plt.yscale('log')
plt.xlabel('h')
plt.ylabel('e1')
plt.title("Erreur en fonction du pas h")
plt.legend()
plt.show()