# -*- coding: utf-8 -*-
"""
Este ejercicio considera una plancha delgada sometida a tracción en tensión
plana. La plancha es de 60 x 30 mm, con un espesor de 10 mm.
El extremo izquierdo de la plancha está empotrado
y sobre la esquina superior derecha se aplica una carga de 10000 Newtons

Donde:
    E = 70 MPa
    poisson = 0.33
    t = 10 mm


@author: Felix Enzo Garofalo Lanzuisi
"""
import numpy as np
import matplotlib.pyplot as plt
from elementos.isoparametrico_cuad import Isoparametrico_Cuad as cuad

# Variables de entrada
gdl = 8                         # Número de grados de libertad por elemento
N = 4                           # Número de elementos
n_g = 18                        # Grados de libertad globales
t = 10                          # Espesor [m]
E = 7e4                         # Módulo de Young [Pa]
poisson = 0.33                  # Coeficiente de poisson

# Crear los nodos
nodo_1 = [0, 0]
nodo_2 = [0, 15]
nodo_3 = [0, 30]
nodo_4 = [30, 0]
nodo_5 = [30, 15]
nodo_6 = [30, 30]
nodo_7 = [60, 0]
nodo_8 = [60, 15]
nodo_9 = [60, 30]


# Crear los elementos
ele_1 = cuad(1, 1, 4, 5, 2, nodo_1, nodo_4, nodo_5, nodo_2, t, E, poisson)
ele_2 = cuad(2, 2, 5, 6, 3, nodo_2, nodo_5, nodo_6, nodo_3, t, E, poisson)
ele_3 = cuad(3, 4, 7, 8, 5, nodo_4, nodo_7, nodo_8, nodo_5, t, E, poisson)
ele_4 = cuad(4, 5, 8, 9, 6, nodo_5, nodo_8, nodo_9, nodo_6, t, E, poisson)

elementos = [ele_1, ele_2, ele_3, ele_4]

# Establecer vector de carga
F = np.zeros(n_g, dtype=float)
F[17] = -10000 # Newtons

# Ensamblaje de la matriz de rigidez
# Inicializamos en cero la matriz de rigidez global
K_G = np.zeros((n_g, n_g), dtype=float)

# Ensamblamos las matrices de rigidez de cada elemento en la
# Matriz de Rigidez Global
for elemento in elementos:
    k_i = elemento.K
    i = 2 * elemento.i - 2
    j = 2 * elemento.j - 2
    k = 2 * elemento.k - 2
    l = 2 * elemento.l - 2

    K_G[i:(i+2), i:(i+2)] += k_i[0:2, 0:2]
    K_G[i:(i+2), j:(j+2)] += k_i[0:2, 2:4]
    K_G[i:(i+2), k:(k+2)] += k_i[0:2, 4:6]
    K_G[i:(i+2), l:(l+2)] += k_i[0:2, 6:8]

    K_G[j:(j+2), i:(i+2)] += k_i[2:4, 0:2]
    K_G[j:(j+2), j:(j+2)] += k_i[2:4, 2:4]
    K_G[j:(j+2), k:(k+2)] += k_i[2:4, 4:6]
    K_G[j:(j+2), l:(l+2)] += k_i[2:4, 6:8]

    K_G[k:(k+2), i:(i+2)] += k_i[4:6, 0:2]
    K_G[k:(k+2), j:(j+2)] += k_i[4:6, 2:4]
    K_G[k:(k+2), k:(k+2)] += k_i[4:6, 4:6]
    K_G[k:(k+2), l:(l+2)] += k_i[4:6, 6:8]

    K_G[l:(l+2), i:(i+2)] += k_i[6:8, 0:2]
    K_G[l:(l+2), j:(j+2)] += k_i[6:8, 2:4]
    K_G[l:(l+2), k:(k+2)] += k_i[6:8, 4:6]
    K_G[l:(l+2), l:(l+2)] += k_i[6:8, 6:8]

# Establecer condiciones de borde
# Condición de borde homogénea
soportes = [0, 1, 2, 3, 4, 5]         # Índices de los GDL restringidos

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

print("Vector de desplazamiento - U:")
print(U.round(3))
print("--------")
print("")


R = K_G.dot(U)
print("Vector de reacciones - R:")
print(R.round(2))
print("")

# Mostrar esfuerzos
for elemento in elementos:
    i = 2 * elemento.i - 2
    j = 2 * elemento.j - 2
    k = 2 * elemento.k - 2
    l = 2 * elemento.l - 2
    print(i, j, k, l)
    elemento.h_i = U[i]
    elemento.v_i = U[i + 1]
    elemento.h_j = U[j]
    elemento.v_j = U[j + 1]
    elemento.h_k = U[k]
    elemento.v_k = U[k + 1]
    elemento.h_l = U[l]
    elemento.v_l = U[l + 1]
    sigma_i = elemento.calcular_esfuerzos()
    print(f"Vector de tensiones - Sigma (elemento {elemento.n}):")
    print(elemento.sigma.round(1))


######################
# Graficar deformación

# Nodos desplazados
escala = 10
U_e = U * escala
n_1 = np.array(nodo_1) + np.array([U_e[0], U_e[1]])
n_2 = np.array(nodo_2) + np.array([U_e[2], U_e[3]])
n_3 = np.array(nodo_3) + np.array([U_e[4], U_e[5]])
n_4 = np.array(nodo_4) + np.array([U_e[6], U_e[7]])
n_5 = np.array(nodo_5) + np.array([U_e[8], U_e[9]])
n_6 = np.array(nodo_6) + np.array([U_e[10], U_e[11]])
n_7 = np.array(nodo_7) + np.array([U_e[12], U_e[13]])
n_8 = np.array(nodo_8) + np.array([U_e[14], U_e[15]])
n_9 = np.array(nodo_9) + np.array([U_e[16], U_e[17]])

originales = np.array([nodo_1, nodo_2, nodo_3, nodo_4, nodo_5,
                       nodo_6, nodo_7, nodo_8, nodo_9])
desplazados = np.array([n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8, n_9])
lineas = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8],
          [0, 3], [3, 6], [1, 4], [4, 7], [2, 5], [5, 8]]

plt.rcParams["figure.figsize"] = [9, 4.50]
plt.rcParams["figure.autolayout"] = True
plt.axis('off')

# Graficar geometría original
for i, linea in enumerate(lineas):
    punto_a = originales[linea[0]]
    punto_b = originales[linea[1]]
    x_a, x_b = punto_a[0], punto_b[0]
    y_a, y_b = punto_a[1], punto_b[1]
    x = [x_a, x_b]
    y = [y_a, y_b]
    plt.plot(x, y, linestyle="-", linewidth=1, color="silver")

# Graficar geometría deformada
for i, linea in enumerate(lineas):
    punto_a = desplazados[linea[0]]
    punto_b = desplazados[linea[1]]
    x_a, x_b = punto_a[0], punto_b[0]
    y_a, y_b = punto_a[1], punto_b[1]
    x = [x_a, x_b]
    y = [y_a, y_b]
    plt.plot(x, y, 'bo', linestyle="--", linewidth=3)
    plt.text(x_a+0.55, y_a-2, f"{linea[0]+1}")
    plt.text(x_b+0.55, y_b-2, f"{linea[1]+1}")

plt.show()



