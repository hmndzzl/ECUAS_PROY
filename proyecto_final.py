
"""
Proyecto Final - Métodos Numéricos
Implementación de RK2 (Heun) y RK4 para:
1) Ecuación Diferencial de Segundo Orden no homogénea:
       y'' + 4y = cos(1.5 t)
   Convertida a sistema y comparada con solución analítica.

2) Péndulo no lineal forzado:
       theta' = omega
       omega' = -(g/L) sin(theta) - k*omega + A*cos(Omega*t)

Incluye:
- Métodos RK2 y RK4
- Estudio de convergencia
- Gráficas
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos

# ================
# MÉTODOS NUMÉRICOS
# ================

def rk2_heun(f, t0, y0, h, n_steps):
    t = t0
    y = np.array(y0, dtype=float)
    ts = [t]
    ys = [y.copy()]
    for i in range(n_steps):
        k1 = f(t, y)
        y_euler = y + h * k1
        k2 = f(t + h, y_euler)
        y = y + (h/2.0)*(k1 + k2)
        t += h
        ts.append(t)
        ys.append(y.copy())
    return np.array(ts), np.array(ys)

def rk4(f, t0, y0, h, n_steps):
    t = t0
    y = np.array(y0, dtype=float)
    ts = [t]
    ys = [y.copy()]
    for i in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h/2.0, y + h*k1/2.0)
        k3 = f(t + h/2.0, y + h*k2/2.0)
        k4 = f(t + h, y + h*k3)
        y = y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        t += h
        ts.append(t)
        ys.append(y.copy())
    return np.array(ts), np.array(ys)

# ============================================
# 1) Ecuación de Segundo Orden no homogénea
# ============================================

def f_second_order(t, y):
    y1, y2 = y
    return np.array([y2, -4*y1 + np.cos(1.5 * t)])

A_part = 1.0 / (4 - 1.5**2)

def y_analytical(t):
    C1 = -A_part
    return C1*np.cos(2*t) + A_part*np.cos(1.5*t)

# ============================================
# 2) Péndulo no lineal forzado
# ============================================

g = 9.81
L = 1.0
k = 0.5
A = 1.2
Omega = 2.0

def f_pendulum(t, y):
    theta, omega = y
    return np.array([
        omega,
        - (g/L)*np.sin(theta) - k*omega + A*np.cos(Omega * t)
    ])

# =============
# EJECUCIÓN
# =============

def main():
    # Parámetros del primer caso
    t0 = 0
    T = 10
    y0 = [0, 0]
    h = 0.01
    n = int((T - t0)/h)

    ts4, ys4 = rk4(f_second_order, t0, y0, h, n)
    ts2, ys2 = rk2_heun(f_second_order, t0, y0, h, n)
    y_exact = y_analytical(ts4)

    # Gráfica caso 1
    plt.figure()
    plt.plot(ts4, ys4[:,0], label="RK4")
    plt.plot(ts2, ys2[:,0], label="RK2")
    plt.plot(ts4, y_exact, '--', label="Analítica")
    plt.legend()
    plt.title("y'' + 4y = cos(1.5 t)")
    plt.grid()
    plt.show()

    # Péndulo
    theta0 = 0.5
    omega0 = 0
    y0_p = [theta0, omega0]

    ts_p4, ys_p4 = rk4(f_pendulum, t0, y0_p, h, n)
    ts_p2, ys_p2 = rk2_heun(f_pendulum, t0, y0_p, h, n)

    # Gráficas péndulo
    plt.figure()
    plt.plot(ts_p4, ys_p4[:,0], label="Theta RK4")
    plt.plot(ts_p2, ys_p2[:,0], label="Theta RK2")
    plt.legend()
    plt.grid()
    plt.title("Péndulo forzado")
    plt.show()

    plt.figure()
    plt.plot(ys_p4[:,0], ys_p4[:,1], label="Fase RK4")
    plt.plot(ys_p2[:,0], ys_p2[:,1], label="Fase RK2")
    plt.legend()
    plt.grid()
    plt.title("Retrato de fase - péndulo")
    plt.show()

if __name__ == "__main__":
    main()
