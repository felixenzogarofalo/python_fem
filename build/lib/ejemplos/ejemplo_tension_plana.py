# -*- coding: utf-8 -*-
"""
Este ejercicio considera una plancha delgada sometida a tracción en tensión
plana. La plancha es de 10 pulgadas por 20 pulgadas, con un espesor de
una pulgada. El extremo izquierdo de la plancha está empotrado
y sobre el extremo derecho se aplica una carga distribuida de 1000 psi

Donde:
    E = 30e6 psi
    poisson = 0.3
    t = 1 in


@author: Felix Enzo Garofalo Lanzuisi
"""
import numpy as np
from elemento_tdc import Elemento_TDC


# Variables de entrada
gdl = 6                         # Número de grados de libertad por elemento
N = 2                           # Número de elementos
n_g = 8                         # Grados de libertad globales
t = 1                           # Espesor [in]
E = 30e6                        # Módulo de Young [psi]
poisson = 0.3                   # Coeficiente de poisson

# Crear los elementos
ele_1 = Elemento_TDC(1, 1, 3, 2, [0, 0], [20, 10], [0, 10], t, E, poisson)
ele_2 = Elemento_TDC(2, 1, 4, 3, [0, 0], [20, 0], [20,10], t, E, poisson)

elementos = [ele_1, ele_2]

# Establecer vector de carga
F = np.zeros(n_g, dtype=float)
F[4] = 5000 # libras
F[6] = 5000 # libras

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
    K_G[i:(i+2), i:(i+2)] += k_i[0:2, 0:2]
    K_G[i:(i+2), j:(j+2)] += k_i[0:2, 2:4]
    K_G[i:(i+2), k:(k+2)] += k_i[0:2, 4:6]
    K_G[j:(j+2), i:(i+2)] += k_i[2:4, 0:2]
    K_G[j:(j+2), j:(j+2)] += k_i[2:4, 2:4]
    K_G[j:(j+2), k:(k+2)] += k_i[2:4, 4:6]
    K_G[k:(k+2), i:(i+2)] += k_i[4:6, 0:2]
    K_G[k:(k+2), j:(j+2)] += k_i[4:6, 2:4]
    K_G[k:(k+2), k:(k+2)] += k_i[4:6, 4:6]

# Establecer condiciones de borde
# Condición de borde homogénea
soportes = [0, 1, 2, 3]         # Índices de los GDL restringidos

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
print(U.round(7)*1e6)
print("--------")
print("")


R = K_G.dot(U)
print("Vector de reacciones - R:")
print(R.round(2))
print("")

for elemento in elementos:
    i = 2 * elemento.i - 2
    j = 2 * elemento.j - 2
    k = 2 * elemento.k - 2
    print(i, j, k)
    elemento.h_1 = U[i]
    elemento.v_1 = U[i+1]
    elemento.h_2 = U[j]
    elemento.v_2 = U[j+1]
    elemento.h_3 = U[k]
    elemento.v_3 = U[k+1]
    sigma_i = elemento.calcular_esfuerzos()
    print(f"Vector de tensiones - Sigma (elemento {elemento.n}):")
    print(elemento.sigma.round(1))





