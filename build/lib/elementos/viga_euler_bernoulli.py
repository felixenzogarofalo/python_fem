# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:32:17 2022

@author: Felix Enzo Garofalo Lanzuisi
"""
import numpy as np
from matplotlib import pyplot as plt

class VigaEB:
    """Esta clase implementa un elemento tipo viga que está sujeto a cargas
    transversales que producen efectos de flexión significativos, pero que
    no toma en cuenta efectos de torsión y/o deformación axial.

    La ecuación diferencial que gobierna el comportamiento de una viga
    elástica lineal elemental, también llamada viga Euler-Bernoulli por sus
    desarrolladores Euler y Bernoulli, se baja en que las secciones planas
    perpendiculares al eje longitudinal permanecerán planas y perpendiculares
    al eje luego de ocurrir la flexión.

    Este es un elemento unidimensional con dos nodos y dos grados de libertad
    por nodo: Un desplazamiento transversal y una rotación por nodo.

    Parámetros:
        float k: constante de elasticidad del resorte
        int n: número del elemento
        int i: número de nodo i
        int j: número de nodo j
        list i_node: coordenadas del punto i
        list j_node: coordenadas del punto j
        float E: módulo de Young
        float I: momento de inercia
    """
    def __init__(self, n, i, j, i_node, j_node, E, I):
        self.n = n
        self.i = i
        self.j = j
        self.i_node = np.array(i_node)
        self.j_node = np.array(j_node)
        L = np.linalg.norm(self.i_node - self.j_node)
        self.L = L
        self.E = E
        self.I = I

        # Definición de la matriz de rigidez
        self.K = (E * I / L**3) * np.array([[12,       6*L,  -12,      6*L],
                                            [6*L, 4*(L**2), -6*L, 2*(L**2)],
                                            [-12,     -6*L,   12,     -6*L],
                                            [6*L, 2*(L**2), -6*L, 4*(L**2)]])

        # Definición de variables de estado internas
        gdl = 4                # Grados de libertad en el elemento
        self.f = np.zeros(gdl)
        self.v_i = 0           # Desplazamiento vertical en nodo i
        self.v_j = 0           # Rotación en nodo i
        self.r_i = 0           # Desplazamiento vertical en nodo j
        self.r_j = 0           # Rotación en nodo j


    @property
    def fuerzas_internas(self):
        """
        Obtener fuerzas internas en el elemento.

        return:
            numpy.array: fuerzas internas al elemento
        """
        u = np.array([self.v_i, self.r_i, self.v_j, self.r_j])
        self.f = self.K.dot(u)
        return self.f

    def graficar_resultados(self):
        self.extraer_dmv()
        x = self.x
        y = self.y
        M = self.M
        V = self.V

        base_line = np.zeros(20)

        plt.style.use("seaborn-whitegrid")

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10, 9))

        ax1.plot(x, base_line, linewidth=1, color="k")
        ax1.plot(x, y, label="Deformada", linewidth=3)
        ax1.set_ylabel("Desplazamiento [in]")
        ax1.set_title("DEFORMADA")
        ax1.fill_between(x, base_line, y, alpha=0.3)

        ax2.plot(x, base_line, linewidth=1, color="k")
        V = np.ones(20) * V
        ax2.plot(x, V, label="Corte", linewidth=3, color="mediumturquoise")
        ax2.set_ylabel("Corte [lbs]")
        ax2.set_title("CORTE")
        ax2.fill_between(x, base_line, V, alpha=0.3, color="mediumturquoise")

        ax3.plot(x, base_line, linewidth=1, color="k")
        ax3.plot(x, M / 12, label="Corte", linewidth=3, color="sandybrown")
        ax3.set_ylabel("Momento [lbs-ft]]")
        ax3.set_title("MOMENTO")
        ax3.fill_between(x, base_line, M/12, alpha=0.3, color="sandybrown")
        ax3.invert_yaxis()

        plt.show()

    def extraer_dmv(self):
        "Devuelve arrays de Numpy con los valores de flexión, momento y corte"

        a_1 = 2 / (self.L**3) * (self.v_i - self.v_j)
        a_1 += 1 / (self.L**2) * (self.r_i + self.r_j)

        a_2 = -3 / (self.L**2) * (self.v_i - self.v_j)
        a_2 += -1 / (self.L) * (2 * self.r_i + self.r_j)

        a_3 = self.r_i

        a_4 = self.v_i

        x = np.linspace(0, self.L, 20)
        self.x = x

        self.y = a_1 * x**3 + a_2 * x**2 + a_3 * x + a_4
        self.M = self.E * self.I * (6 * a_1 * x + 2 * a_2)
        V = self.E * self.I * (6 * a_1)
        self.V = np.ones(20) * V

        return np.array([self.y, self.M, self.V]).transpose()





