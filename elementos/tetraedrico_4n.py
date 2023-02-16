# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 08:56:33 2023

@author: Felix Enzo Garofalo Lanzuisi
"""

import numpy as np

class Tetraedro_4n:
    """ Esta clase implementa un elemento tetraédrico lineal de cuatro
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
                 E, poisson):
        self.n = n
        self.i = i
        self.j = j
        self.k = k
        self.l = l
        self.i_node = np.array(i_node)
        self.j_node = np.array(j_node)
        self.k_node = np.array(k_node)
        self.l_node = np.array(l_node)
        self.E = E
        self.poisson = poisson

        # Variables de coordenadas
        self.x1, self.y1, self.z1 = i_node[0], i_node[1], i_node[2]
        self.x2, self.y2, self.z2 = j_node[0], j_node[1], j_node[2]
        self.x3, self.y3, self.z3 = k_node[0], k_node[1], k_node[2]
        self.x4, self.y4, self.z4 = l_node[0], l_node[1], l_node[2]

        # Variables a ser calculadas
        self.h_i = 0
        self.h_j = 0
        self.h_k = 0
        self.h_l = 0
        self.v_i = 0
        self.v_j = 0
        self.v_k = 0
        self.v_l = 0

        # Determinar 6V
        seis_V = np.array([[1, self.x1, self.y1, self.z1],
                           [1, self.x2, self.y2, self.z2],
                           [1, self.x3, self.y3, self.z3],
                           [1, self.x4, self.y4, self.z4]])
        seis_V = np.linalg.det(seis_V)
        self.seis_V = seis_V
        self.volumen = seis_V / 6

        # Factores alfa_i, beta_i, gamma_i y delta_i para la
        # Matriz Gradiente B

        # Alfa 1
        alfa_1 = np.array([[self.x2, self.y2, self.z2],
                           [self.x3, self.y3, self.z3],
                           [self.x4, self.y4, self.z4]])
        alfa_1 = np.linalg.det(alfa_1)
        self.alfa_1 = alfa_1

        # Beta 1
        beta_1 = np.array([[1, self.y2, self.z2],
                           [1, self.y3, self.z3],
                           [1, self.y4, self.z4]])
        beta_1 = -np.linalg.det(beta_1)
        self.beta_1 = beta_1

        # Gamma 1
        gamma_1 = np.array([[1, self.x2, self.z2],
                            [1, self.x3, self.z3],
                            [1, self.x4, self.z4]])
        gamma_1 = np.linalg.det(gamma_1)
        self.gamma_1 = gamma_1

        # Delta 1
        delta_1 = np.array([[1, self.x2, self.y2],
                            [1, self.x3, self.y3],
                            [1, self.x4, self.y4]])
        delta_1 = -np.linalg.det(delta_1)
        self.delta_1 = delta_1

        # Alfa 2
        alfa_2 = np.array([[self.x1, self.y1, self.z1],
                           [self.x3, self.y3, self.z3],
                           [self.x4, self.y4, self.z4]])
        alfa_2 = -np.linalg.det(alfa_2)
        self.alfa_2 = alfa_2

        # Beta 2
        beta_2 = np.array([[1, self.y1, self.z1],
                           [1, self.y3, self.z3],
                           [1, self.y4, self.z4]])
        beta_2 = np.linalg.det(beta_2)
        self.beta_2 = beta_2

        # Gamma 2
        gamma_2 = np.array([[1, self.x1, self.z1],
                            [1, self.x3, self.z3],
                            [1, self.x4, self.z4]])
        gamma_2 = -np.linalg.det(gamma_2)
        self.gamma_2 = gamma_2

        # Delta 2
        delta_2 = np.array([[1, self.x1, self.y1],
                            [1, self.x3, self.y3],
                            [1, self.x4, self.y4]])
        delta_2 = np.linalg.det(delta_2)
        self.delta_2 = delta_2

        # Alfa 3
        alfa_3 = np.array([[self.x1, self.y1, self.z1],
                           [self.x3, self.y3, self.z3],
                           [self.x4, self.y4, self.z4]])
        alfa_3 = np.linalg.det(alfa_3)
        self.alfa_3 = alfa_3

        # Beta 3
        beta_3 = np.array([[1, self.y1, self.z1],
                           [1, self.y2, self.z2],
                           [1, self.y4, self.z4]])
        beta_3 = -np.linalg.det(beta_3)
        self.beta_3 = beta_3

        # Gamma 3
        gamma_3 = np.array([[1, self.x1, self.z1],
                            [1, self.x2, self.z2],
                            [1, self.x4, self.z4]])
        gamma_3 = np.linalg.det(gamma_3)
        self.gamma_3 = gamma_3

        # Delta 3
        delta_3 = np.array([[1, self.x1, self.y1],
                            [1, self.x2, self.y2],
                            [1, self.x4, self.y4]])
        delta_3 = -np.linalg.det(delta_3)
        self.delta_3 = delta_3

        # Alfa 4
        alfa_4 = np.array([[self.x1, self.y1, self.z1],
                           [self.x3, self.y3, self.z3],
                           [self.x3, self.y3, self.z3]])
        alfa_4 = -np.linalg.det(alfa_4)
        self.alfa_4 = alfa_4

        # Beta 4
        beta_4 = np.array([[1, self.y1, self.z1],
                           [1, self.y2, self.z2],
                           [1, self.y3, self.z3]])
        beta_4 = np.linalg.det(beta_4)
        self.beta_4 = beta_4

        # Gamma 4
        gamma_4 = np.array([[1, self.x1, self.z1],
                            [1, self.x2, self.z2],
                            [1, self.x3, self.z3]])
        gamma_4 = -np.linalg.det(gamma_4)
        self.gamma_4 = gamma_4

        # Delta 4
        delta_4 = np.array([[1, self.x1, self.y1],
                            [1, self.x2, self.y2],
                            [1, self.x3, self.y3]])
        delta_4 = np.linalg.det(delta_4)
        self.delta_4 = delta_4

        self.factores = np.array([[alfa_1, beta_1, gamma_1, delta_1],
                                  [alfa_2, beta_2, gamma_2, delta_2],
                                  [alfa_3, beta_3, gamma_3, delta_3],
                                  [alfa_4, beta_4, gamma_4, delta_4]])

        # Funciones de forma * Solo de referencia
        # N_i = 1 / 6V * (alfa_i + beta_i * x + gamma_i * y + delta_i * z)


        # Derivadas parciales de funciones de forma
        N_1x = self.beta_1 / self.seis_V
        N_1y = self.gamma_1 / self.seis_V
        N_1z = self.delta_1 / self.seis_V
        N_2x = self.beta_2 / self.seis_V
        N_2y = self.gamma_2 / self.seis_V
        N_2z = self.delta_2 / self.seis_V
        N_3x = self.beta_3 / self.seis_V
        N_3y = self.gamma_3 / self.seis_V
        N_3z = self.delta_3 / self.seis_V
        N_4x = self.beta_4 / self.seis_V
        N_4y = self.gamma_4 / self.seis_V
        N_4z = self.delta_4 / self.seis_V

        N_ij = np.array([[N_1x, N_1y, N_1z],
                         [N_2x, N_2y, N_2z],
                         [N_3x, N_3y, N_3z],
                         [N_4x, N_4y, N_4z]])

        self.N_ij = N_ij

        # Definición de matriz constitutiva
        D = np.array([[1-poisson, poisson, poisson, 0, 0, 0],
                      [poisson, 1-poisson, poisson, 0, 0, 0],
                      [poisson, poisson, 1-poisson, 0, 0, 0],
                      [0, 0, 0, (1 - 2 * poisson) / 2, 0, 0],
                      [0, 0, 0, 0, (1 - 2 * poisson) / 2, 0],
                      [0, 0, 0, 0, 0, (1 - 2 * poisson) / 2]])
        D = E / ((1 + poisson) * (1 - 2 * poisson)) * D
        self.D = D

        # Definición de Matriz Gradiente
        B = np.hstack((self.B_i(0), self.B_i(1), self.B_i(2), self.B_i( 3)))
        self.B = B

        # Cómputo de Matriz de Rigidez
        K = np.dot(np.dot(B.transpose(),D),B) * self.volumen
        self.K = K

    def B_i(self, i):
        B_i = np.zeros((6, 3))
        B_i[0, 0] = self.factores[i, 1]
        B_i[1, 1] = self.factores[i, 2]
        B_i[2, 2] = self.factores[i, 3]
        B_i[3, 0] = self.factores[i, 2]
        B_i[3, 1] = self.factores[i, 1]
        B_i[4, 1] = self.factores[i, 3]
        B_i[4, 2] = self.factores[i, 2]
        B_i[5, 0] = self.factores[i, 3]
        B_i[5, 2] = self.factores[i, 1]
        B_i = 1 / self.seis_V * B_i
        return B_i

    # Determinar esfuerzos en punto medio
    def calcular_esfuerzos(self):
        B = self.B()
        q = np.array([self.h_i, self.v_i,
                      self.h_j, self.v_j,
                      self.h_k, self.v_k,
                      self.h_l, self.v_l])
        sigma = np.dot(np.dot(self.D, B), q)
        self.sigma = sigma
        return sigma

    def __repr__(self):
        return f"Elemento {self.n}: {self.i}-{self.j}-{self.k}-{self.l}"
