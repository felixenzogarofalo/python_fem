# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 08:56:33 2023

@author: Felix Enzo Garofalo Lanzuisi
"""

import numpy as np

class Isoparametrico_Cuad:
    """ Esta clase implementa un elemento isoparamétrico lineal de cuatro
    nodos.

    Parámetros:
        float k: constante de elasticidad del resorte
        int n: número del elemento
        int i: número del nodo i
        int j: número del nodo j
        int k: número del nodo k
        int l: número del nodo l
        list i_node: coordenadas del punto i
        list j_node: coordenadas del punto j
        list k_node: coordenadas del punto k
        list l: coordenadas del punto l
        float t: espesor
        float E: módulo de Young
        float poisson: coeficiente de Poisson
        int modo: '0' para esfuerzo plano y '1' para deformación plana
    """
    def __init__(self, n, i, j, k, l, i_node, j_node, k_node, l_node,
                 t, E,poisson, modo=0):
        self.n = n
        self.i = i
        self.j = j
        self.k = k
        self.l = l
        self.i_node = np.array(i_node)
        self.j_node = np.array(j_node)
        self.k_node = np.array(k_node)
        self.l_node = np.array(l_node)
        self.t = t
        self.E = E
        self.poisson = poisson

        # Variables a ser calculadas
        self.h_i = 0
        self.h_j = 0
        self.h_k = 0
        self.h_l = 0
        self.v_i = 0
        self.v_j = 0
        self.v_k = 0
        self.v_l = 0

        # Funciones de forma
        N_1 = lambda s, t: 1 / 4 *(1 - s) * (1 - t)
        N_2 = lambda s, t: 1 / 4 *(1 + s) * (1 - t)
        N_3 = lambda s, t: 1 / 4 *(1 + s) * (1 + t)
        N_4 = lambda s, t: 1 / 4 *(1 - s) * (1 + t)

        # Derivadas parciales de funciones de forma
        N_1s = lambda s, t: 1 / 4 * (t - 1)
        N_1t = lambda s, t: 1 / 4 * (s - 1)
        N_2s = lambda s, t: 1 / 4 * (1 - t)
        N_2t = lambda s, t: -1 / 4 * (1 + s)
        N_3s = lambda s, t: 1 / 4 * (1 + t)
        N_3t = lambda s, t: 1 / 4 * (1 + s)
        N_4s = lambda s, t: -1 / 4 * (1 + t)
        N_4t = lambda s, t: 1 / 4 * (1 - s)

        N_ii = np.array([N_1s, N_1t],
                        [N_2s, N_2t],
                        [N_3s, N_3t],
                        [N_4s, N_4t])

        # Factores a, b, c y de para matriz gradiente "B"
        def a(s, t):
            a  = 1 / 4 * (i_node[1] * (s - 1) + j_node[1] * (-1 - s))
            a += 1 / 4 * (k_node[1] * (1 + s) + l_node[1] * (1 - s))
            return a

        def b(s, t):
            b  = 1 / 4 * (i_node[1] * (t - 1) + j_node[1] * (1 - t))
            b += 1 / 4 * (k_node[1] * (1 + t) + l_node[1] * (-1 - t))
            return b

        def c(s, t):
            c  = 1 / 4 * (i_node[0] * (t - 1) + j_node[0] * (1 - t))
            c += 1 / 4 * (k_node[0] * (1 + t) + l_node[0] * (-1 - t))
            return c

        def d(s, t):
            d  = 1 / 4 * (i_node[0] * (s - 1) + j_node[0] * (-1 - s))
            d += 1 / 4 * (k_node[0] * (1 + s) + l_node[0] * (1 - s))
            return d

        def B_i(s, t, i):
            B_i = np.zeros((3, 2))
            B_i[0, 0] = a(s, t) * N_ii[i,0](s, t) - b(s, t) * N_ii[i,1](s, t)
            B_i[1, 1] = c(s, t) * N_ii[i,1](s, t) - d(s, t) * N_ii[i,0](s. t)
            B_i[2, 0] = c(s, t) * N_ii[i,1](s, t) - d(s, t) * N_ii[i,0](s. t)
            B_i[2, 1] = a(s, t) * N_ii[i,0](s, t) - b(s, t) * N_ii[i,1](s, t)

        def jacobiano(s, t):
            X_c = [self.i_node[0], self.j_node[0], self.k_node[0], self.l_node[0]]
            Y_c = [self.i_node[1], self.j_node[1], self.k_node[1], self.l_node[1]]
            jacobiano = np.array([0, 1 - t, t - s, s - 1],
                                 [t - 1, 0, s + 1, -s - 1],
                                 [s - t, -s - 1, 0, t +1],
                                 [1 - s, s + 1, -t - 1, 0])
            jacobiano = 1 / 8 * X_c.transpose() * jacobiano * Y_c
            return jacobiano

        # Matriz gradiente
        def B(s, t):
            1 / jacobiano(s, t) * np.array([B_i(s, t, 0), B_i(s, t, 1), B_i(s, t, 2), B_i(s, t, 3)])

        # Definición de matriz constitutiva
        # Caso de deformación plana
        if modo == 1:
            factor = E / ((1 + poisson)*(1 - 2*poisson))
            D = factor * np.array([[1-poisson, poisson, 0],
                                   [poisson, 1-poisson, 0],
                                   [0, 0, (1-2*poisson)/2]],
                                  dtype=float)
        # Caso de esfuerzo plano
        else:
            factor = E / (1 - poisson**2)
            D = factor * np.array([[1, poisson, 0],
                                   [poisson, 1, 0],
                                   [0, 0, (1 - poisson)/2]],
                                  dtype=float)
        self.D = D

        # Integración numérica de matriz de rigidez
        # Cuadratura de Gauss para 2 puntos por dirección
        K = np.zeros((8, 8))

        s = [-.5773, -.5773, .5773, .5773]
        t = [-.5773, .5773, -.5773, .5773]
        W = [1, 1, 1, 1]

        for i in range(4):
            s = s[i]
            t = t[i]
            W = W[i]

            jacobiano = jacobiano(s, t)
            B = B(s, t)
            D = self.D
            K += t * np.matmul(np.matmul(B.transpose(),D),B) * jacobiano * W * W

        self.K = K
