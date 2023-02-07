# -*- coding: utf-8 -*-
"""
Este ejercicio considera cuatro vigas Euler-Bernoulli unidas entre si según la
siguiente configuración:

                 (1)            (2)            (3)            (4)
          1              2              3              4              5
          o--------------o--------------o--------------o--------------o
          |                             ^                             |
                 L               L              L              L

Con cargas puntuales aplicadas en dirección Y negativa para los nodos
2 y 4 de magnitud 10,000 lbs

Donde:
    L = 10 ft ó 120 in
    E = 30e6 psi
    I = 500 in^4
    F = 10.000 lb


@author: Felix Enzo Garofalo Lanzuisi
"""
import numpy as np
from elementos.viga_euler_bernoulli import VigaEB
from matplotlib import pyplot as plt

# Variables de entrada
gdl = 4                         # Número de grados de libertad por elemento
N = 2                           # Número de elementos
n_g = 10                        # Grados de libertad globales
E = 30e6                        # Módulo de Young [psi]
I = 500                         # Momento de inercia [in^4]
L = 10*12                       # Lontitud [in]

# Crear los elementos
viga_1 = VigaEB(1, 1, 2, [0,0,0], [120,0,0], E, I)
viga_2 = VigaEB(2, 2, 3, [120,0,0], [240,0,0], E, I)
viga_3 = VigaEB(3, 3, 4, [240,0,0], [360,0,0], E, I)
viga_4 = VigaEB(4, 4, 5, [360,0,0], [480,0,0], E, I)


elementos = [viga_1, viga_2, viga_3, viga_4]

# Establecer vector de carga
F = np.zeros(n_g)
F[2] = -10000 # libras
F[6] = -10000 # libras

# Ensamblaje de la matriz de rigidez
# Inicializamos en cero la matriz de rigidez global
K_G = np.zeros((n_g, n_g))

# Ensamblamos las matrices de rigidez de cada elemento en la
# Matriz de Rigidez Global
for elemento in elementos:
    k_i = elemento.K
    i = 2 * elemento.i - 2
    j = 2 * elemento.j - 2
    K_G[i:(i+2), i:(i+2)] += k_i[0:2, 0:2]
    K_G[i:(i+2), j:(j+2)] += k_i[0:2, 2:4]
    K_G[j:(j+2), i:(i+2)] += k_i[2:4, 0:2]
    K_G[j:(j+2), j:(j+2)] += k_i[2:4, 2:4]


# Establecer condiciones de borde
# Condición de borde homogénea
soportes = [0, 1, 4, 8, 9]         # Índices de los GDL restringidos

# Reducir Matriz de Rigidez Global y vector de fuerza
K_R = K_G
F_R = F
K_R = np.delete(K_R, soportes, 0)
K_R = np.delete(K_R, soportes, 1)
F_R = np.delete(F_R, soportes, 0)

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
print(U.round(3))
print("--------")
print("")


R = K_G.dot(U)
print("Vector de reacciones - R:")
print(R.round(2))
print("")

# Obtener fuerzas internas
for elemento in elementos:
    elemento.v_i = U[2 * elemento.i - 2]
    elemento.r_i = U[2 * elemento.i - 1]
    elemento.v_j = U[2 * elemento.j - 2]
    elemento.r_j = U[2 * elemento.j - 1]
    print(f"Fuerzas Internas de elemento {elemento.n}:")
    print(elemento.fuerzas_internas.round(2))

# Graficar el tramo 2
viga_2.graficar_resultados()

# Graficar todos los tramos
res = []
for viga in elementos:
    if len(res) == 0:
        res = viga.extraer_dmv()
    else:
        res = np.vstack((res, viga.extraer_dmv()))

y = res[:, 0]
V = res[:, 2]
M = res[:, 1]

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10, 9))
base_line = np.zeros(len(res))
x = np.linspace(0, L*4, 80)
ax1.plot(x, base_line, linewidth=1, color="k")
ax1.plot(x, y, label="Deformada", linewidth=3)
ax1.set_ylabel("Desplazamiento [in]")
ax1.set_title("DEFORMADA")
ax1.fill_between(x, base_line, res[:,0], alpha=0.3)

ax2.plot(x, base_line, linewidth=1, color="k")
V = np.ones(len(res)) * V
ax2.plot(x, V, label="Corte", linewidth=3, color="mediumturquoise")
ax2.set_ylabel("Corte [lbs]")
ax2.set_title("CORTE")
ax2.fill_between(x, base_line, V, alpha=0.3, color="mediumturquoise")

ax3.plot(x, base_line, linewidth=1, color="k")
ax3.plot(x, M / 12, label="Corte", linewidth=3, color="sandybrown")
ax3.set_ylabel("Momento [lbs-ft]]")
ax3.set_title("MOMENTO")
ax3.fill_between(x, base_line, M / 12, alpha=0.3, color="sandybrown")
ax3.invert_yaxis()

plt.show()
