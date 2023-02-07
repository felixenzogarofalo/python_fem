# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:32:17 2022

@author: Felix Enzo Garofalo Lanzuisi
"""
import numpy as np

class Elemento_TDC:
    """Esta clase implementa un elemento triangular de deformación constante.


    Parámetros:
        float k: constante de elasticidad del resorte
        int n: número del elemento
        int i: número de nodo i
        int j: número de nodo j
        int k: número de nodo k
        list i_node: coordenadas del punto i
        list j_node: coordenadas del punto j
        list k_node: coordenadas del punto k
        float t: espesor
        float E: módulo de Young
        float poisson: coeficiente de Poisson
        int modo: '0 para esfuerzo plano y '1' para deformación plana
    """
    def __init__(self, n, i, j, k, i_node, j_node, k_node, t, E, poisson, modo=0):
        self.n = n
        self.i = i
        self.j = j
        self.k = k
        self.i_node = np.array(i_node)
        self.j_node = np.array(j_node)
        self.k_node = np.array(k_node)
        vector_ij = self.j_node - self.i_node
        vector_ik = self.k_node - self.i_node
        producto_ij_ik = np.cross(vector_ij, vector_ik)
        area = np.linalg.norm(producto_ij_ik) / 2
        self.area = area
        self.t = t
        self.E = E
        self.poisson = poisson

        # Variables a ser calculadas
        self.h_1 = 0
        self.h_2 = 0
        self.h_3 = 0
        self.v_1 = 0
        self.v_2 = 0
        self.v_3 = 0

        # Definición de Matriz B
        x_i = i_node[0]
        x_j = j_node[0]
        x_k = k_node[0]
        y_i = i_node[1]
        y_j = j_node[1]
        y_k = k_node[1]

        beta_i = y_j - y_k
        beta_j = y_k - y_i
        beta_k = y_i - y_j
        gamma_i = x_k - x_j
        gamma_j = x_i - x_k
        gamma_k = x_j - x_i

        B = (1/(2*area)) * np.array([[beta_i, 0, beta_j, 0, beta_k, 0],
                                     [0, gamma_i, 0, gamma_j, 0, gamma_k],
                                     [gamma_i, beta_i, gamma_j, beta_j, gamma_k, beta_k]],
                                    dtype=float)
        self.B = B

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

        # Definición de la matriz de rigidez
        self.K = t * area * np.matmul(np.matmul(B.transpose(),D),B)

    def calcular_esfuerzos(self):
        d = np.array([self.h_1, self.v_1, self.h_2, self.v_2, self.h_3, self.v_3])
        self.d = d
        sigma = np.matmul(self.D, self.B)
        sigma = np.matmul(sigma,d.transpose())
        self.sigma = sigma
        return self.sigma
