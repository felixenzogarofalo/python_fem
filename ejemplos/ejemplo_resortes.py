# -*- coding: utf-8 -*-
"""
Este ejercicio considera dos resortes unidos entre si según la siguiente
configuración:

                 k_1            k_2
          1              2              3
          o--------------o--------------o -> F_3
          |              ^              ^

Donde:

    k_1: constante de resorte 1 = 1000 [kN/m]
    k_2: constante de resorte 2 = 1000 [kN/m]
    F_3: fuerza puntual en nodo 3 = 500 kN

@author: Felix Enzo Garofalo Lanzuisi
"""
import numpy as np
from elementos.resorte import CrearResorte

# Variables de entrada
gdl = 2                         # Número de grados de libertad por elemento
N = 2                           # Número de elementos
n_g = gdl * N - (N-1)           # Grados de libertad globales
k_1 = 1000                      # k del resorte 1 [kN/m]
k_2 = 1000                      # k del resorte 2 [kN/m]

# Crear los elementos
resorte_1 = CrearResorte(k_1, 1, 1, 2)
resorte_2 = CrearResorte(k_2, 2, 2, 3)

elementos = [resorte_1, resorte_2]

# Establecer vector de carga
F = np.zeros(n_g)
F[2] = 500

# Ensamblaje de la matriz de rigidez
# Inicializamos en cero la matriz de rigidez global
K_G = np.zeros((n_g, n_g))

# Ensamblamos las matrices de rigidez de cada elemento en la
# Matriz de Rigidez Global
for elemento in elementos:
    k_i = elemento.K
    nodo_i = elemento.i - 1
    nodo_j = elemento.j - 1
    K_G[nodo_i, nodo_i] += k_i[0, 0]
    K_G[nodo_i, nodo_j] += k_i[0, 1]
    K_G[nodo_j, nodo_i] += k_i[1, 0]
    K_G[nodo_j, nodo_j] += k_i[1, 1]


# Establecer condiciones de borde
# Condición de borde homogénea
soportes = [0]                       # Índices de los GDL restringidos

# Reducir Matriz de Rigidez Global y vector de fuerza
K_R = K_G
F_R = F
for restriccion in soportes:
    K_R = np.delete(K_R, restriccion, 0)
    K_R = np.delete(K_R, restriccion, 1)
    F_R = np.delete(F_R, restriccion, 0)

# Resolvemos los desplazamientos
U_R = np.linalg.solve(K_R, F_R)

# Expandir vector de desplazamientos
U = np.zeros(n_g)
j = 0
for i in np.arange(n_g):
    if i not in soportes:
        U[i] = U_R[j]
        j += 1

print("Vector de desplazamiento - U:")
print(U.round(2))
print("--------")
print("")

# Obtener vector de fuerzas nodales
F_n = K_G.dot(U)
print("Vector de reacciones - R:")
print(F_n.round(2))
print("")

# Obtener fuerzas internas
for elemento in elementos:
    elemento.u_i = U[elemento.i - 1]
    elemento.u_j = U[elemento.j - 1]
    print(f"Fuerzas Internas de elemento {elemento.n}:")
    print(elemento.fuerzas_internas.round(2))

