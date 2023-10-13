# -*- coding: utf-8 -*-
"""
Este ejercicio considera soporte en catiliever de 50 x 100 x 50
El extremo izquierdo del soporte está empotrado y se aplica una carga de
1000 newtons en los nodos superiores derechos

Donde:
    E = 70 MPa
    poisson = 0.33
    t = 10 mm


@author: Felix Enzo Garofalo Lanzuisi
"""
import matplotlib.pyplot as plt
import numpy as np
from tetraedrico_4n import Tetraedro_4n as tetra

# Variables de entrada
gdl = 8                         # Número de grados de libertad por elemento
N = 10                          # Número de elementos
n_g = 36                        # Grados de libertad globales
E = 7e4                         # Módulo de Young [Pa]
poisson = 0.33                  # Coeficiente de poisson

# Crear los nodos
nodo_1 = [0, 0, 0]
nodo_2 = [50, 0, 0]
nodo_3 = [50, 50, 0]
nodo_4 = [0, 50, 0]
nodo_5 = [0, 0, 50]
nodo_6 = [50, 0, 50]
nodo_7 = [50, 50, 50]
nodo_8 = [0, 50, 50]
nodo_9 = [100, 0, 0]
nodo_10 = [100, 50, 0]
nodo_11 = [100, 0, 50]
nodo_12 = [100, 50, 50]

nodos = [nodo_1, nodo_2, nodo_3, nodo_4, nodo_5, nodo_6,
         nodo_7, nodo_8, nodo_9, nodo_10, nodo_11, nodo_12]


# Crear elemeentos
ele_1 = tetra(1, 1, 2, 4, 5, nodo_1, nodo_2, nodo_4, nodo_5, E, poisson)
ele_2 = tetra(2, 2, 4, 5, 7, nodo_2, nodo_4, nodo_5, nodo_7, E, poisson)
ele_3 = tetra(3, 2, 3, 4, 7, nodo_2, nodo_3, nodo_4, nodo_7, E, poisson)
ele_4 = tetra(4, 1, 6, 5, 8, nodo_1, nodo_6, nodo_5, nodo_8, E, poisson)
ele_5 = tetra(5, 4, 5, 7, 8, nodo_4, nodo_5, nodo_7, nodo_8, E, poisson)
ele_6 = tetra(6, 2, 3, 6, 9, nodo_2, nodo_3, nodo_6, nodo_9, E, poisson)
ele_7 = tetra(7, 3, 9, 10, 12, nodo_3, nodo_9, nodo_10, nodo_12, E, poisson)
ele_8 = tetra(8, 3, 6, 9, 12, nodo_3, nodo_6, nodo_9, nodo_12, E, poisson)
ele_9 = tetra(9, 3, 6, 7, 12, nodo_3, nodo_6, nodo_7, nodo_12, E, poisson)
ele_10 = tetra(10, 6, 9, 11, 12, nodo_6, nodo_9, nodo_11, nodo_12, E, poisson)

elementos = [ele_1, ele_2, ele_3, ele_4, ele_5,
             ele_6, ele_7, ele_8, ele_9, ele_10]

# Establecer vector de carga
F = np.zeros(n_g, dtype=float)
#F[33] = -1000 # Newtons
F[35] = -1000 # Newtons

# Ensamblaje de la matriz de rigidez
# Inicializamos en cero la matriz de rigidez global
K_G = np.zeros((n_g, n_g), dtype=float)

# Ensamblamos las matrices de rigidez de cada elemento en la
# Matriz de Rigidez Global
for elemento in elementos:
    k_i = elemento.K
    i = 3 * elemento.i - 3
    j = 3 * elemento.j - 3
    k = 3 * elemento.k - 3
    l = 3 * elemento.l - 3

    K_G[i:(i+3), i:(i+3)] += k_i[0:3, 0:3]
    K_G[i:(i+3), j:(j+3)] += k_i[0:3, 3:6]
    K_G[i:(i+3), k:(k+3)] += k_i[0:3, 6:9]
    K_G[i:(i+3), l:(l+3)] += k_i[0:3, 9:12]

    K_G[j:(j+3), i:(i+3)] += k_i[3:6, 0:3]
    K_G[j:(j+3), j:(j+3)] += k_i[3:6, 3:6]
    K_G[j:(j+3), k:(k+3)] += k_i[3:6, 6:9]
    K_G[j:(j+3), l:(l+3)] += k_i[3:6, 9:12]

    K_G[k:(k+3), i:(i+3)] += k_i[6:9, 0:3]
    K_G[k:(k+3), j:(j+3)] += k_i[6:9, 3:6]
    K_G[k:(k+3), k:(k+3)] += k_i[6:9, 6:9]
    K_G[k:(k+3), l:(l+3)] += k_i[6:9, 9:12]

    K_G[l:(l+3), i:(i+3)] += k_i[9:12, 0:3]
    K_G[l:(l+3), j:(j+3)] += k_i[9:12, 3:6]
    K_G[l:(l+3), k:(k+3)] += k_i[9:12, 6:9]
    K_G[l:(l+3), l:(l+3)] += k_i[9:12, 9:12]

# Establecer condiciones de borde
# Condición de borde homogénea
# Índices de los GDL restringidos
soportes = [0, 1, 2, 9, 10, 11, 12, 13, 14, 21, 22, 23]

# Reducir Matriz de Rigidez Global y vector de fuerza
K_R = K_G
F_R = F
K_R = np.delete(K_R, soportes, 0)
K_R = np.delete(K_R, soportes, 1)
F_R = np.delete(F_R, soportes, 0)

# # Resolvemos los desplazamientos
U_R = np.linalg.solve(K_R, F_R)

# Expandir vector de desplazamientos
U = np.zeros(n_g)
j = 0
for i in np.arange(n_g):
    if i not in soportes:
        U[i] = U_R[j]
        j += 1

print("Vector de desplazamiento - U [[mm]]:")
print(U.round(3))
print("--------")
print("")


R = K_G.dot(U)
print("Vector de reacciones - R:")
print(R.round(2))
print("")




##########
# Graficar

# Escalar deformación
escala = 1000
u_d = U.reshape((12,3)) * escala

# Graficar nodos
x = []
y = []
z = []
for i, nodo in enumerate(nodos):
    valores = np.array(nodo) + u_d[i]
    valores = valores.tolist()
    x.append(valores[0])
    y.append(valores[1])
    z.append(valores[2])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x, y, z, marker="o")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()