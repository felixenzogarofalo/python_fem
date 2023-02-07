# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:32:17 2022

@author: User
"""
import numpy as np

class CrearResorte:
    """Esta clase implementa un elemento tipo resorte que solo tiene
    dos grados de libertad: un desplazamiento a lo largo de la dirección
    axial para cada nodo.
    
    Parámetros: 
        float k: constante de elasticidad del resorte
        int n: número del elemento
        int i: número de nodo i
        int j: número de nodo j
    """
    def __init__(self, k, n, i, j):
        self.k = k
        self.n = n
        self.i = i
        self.j = j
        self.K = np.array([[k, -k],
                           [-k, k]])
        self.f = np.zeros(2)
        self.u_i = 0
        self.u_j = 0

    
    @property
    def fuerzas_internas(self):
        """
        Obtener fuerzas internas en el elemento.
        
        Parámetros:
            float u_i: desplazamiento en nodo i
            float u_j: desplazamiento en nodo j
        
        return:
            numpy.array: fuerzas internas al elemento
        """
        u = np.array([self.u_i, self.u_j])
        self.f = self.K.dot(u)
        return self.f
        