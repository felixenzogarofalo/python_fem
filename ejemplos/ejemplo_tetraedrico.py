# -*- coding: utf-8 -*-
"""
Este ejercicio considera un elemento tetraédrico


@author: Felix Enzo Garofalo Lanzuisi
"""
from elementos.tetraedrico_4n import Tetraedro_4n as tetra

# Variables de entrada
gdl = 12                        # Número de grados de libertad por elemento
N = 1                           # Número de elementos
n_g = 12                        # Grados de libertad globales
E = 30e6                        # Módulo de Young [Pa]
poisson = 0.30                  # Coeficiente de poisson

# Crear los nodos
nodo_1 = [1, 1, 2]
nodo_2 = [0, 0, 0]
nodo_3 = [0, 2, 0]
nodo_4 = [2, 1, 0]


# Crear elemeentos
ele = tetra(1, 1, 2, 3, 4, nodo_1, nodo_2, nodo_3, nodo_4, E, poisson)

print(ele.B)

