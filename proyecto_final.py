import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos

# ==================
# MÉTODOS NUMÉRICOS
# ==================

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


# ====================
# ED DE PRIMER ORDEN
# y' + y*tan(x) = cos^2(x)
# ====================

def f_primer_orden(t, y):
    return np.array([np.cos(t)**2 - y[0]*np.tan(t)])

def y_primer_orden(t):
    return np.cos(t)*np.sin(t) + np.cos(t)


def primer_orden():
    print("\n--- ECUACIÓN DE PRIMER ORDEN ---")

    t0 = 0
    T  = 3
    h  = 0.01
    steps = int((T - t0)/h)

    y0 = [1.0]

    ts2, ys2 = rk2_heun(f_primer_orden, t0, y0, h, steps)
    ts4, ys4 = rk4(f_primer_orden, t0, y0, h, steps)
    
    y_ex = y_primer_orden(ts4)

    err2 = np.max(np.abs(ys2[:,0] - y_ex))
    err4 = np.max(np.abs(ys4[:,0] - y_ex))

    print(f"Error RK2 = {err2}")
    print(f"Error RK4 = {err4}")

    return ts2, ys2, ts4, ys4

def graficar_primer_orden(ts2, ys2, ts4, ys4):
    """Genera la gráfica de comparación para la ED de Primer Orden."""
    y_exact = y_primer_orden(ts4)
    plt.figure(figsize=(10, 6))
    plt.plot(ts4, y_exact, 'k--', label="Analítica", linewidth=1.5)
    plt.plot(ts4, ys4[:,0], label="RK4", linewidth=2)
    plt.plot(ts2, ys2[:,0], label="RK2 (Heun)", linestyle=':', linewidth=2)
    
    plt.legend()
    plt.title(r'ED de Primer Orden: $\frac{dy}{dt} = \cos^2(t) - y \tan(t)$')
    plt.xlabel('Tiempo $t$')
    plt.ylabel('Solución $y(t)$')
    plt.grid(True)
    plt.show()


# =================================
# ED DE SEGUNDO ORDEN NO HOMOGÉNEA
# y'' + 4y = x^2
# =================================

def f_segundo_orden(t, y):
    y1, y2 = y
    dy1 = y2
    dy2 = t**2 - 4*y1
    return np.array([dy1, dy2])

def y_segundo_orden(t, y0):
    C1 = y0[0] + 1/8
    C2 = y0[1] / 2
    return C1*np.cos(2*t) + C2*np.sin(2*t) + (t**2)/4 - 1/8

def segundo_orden():
    print("\n--- ECUACIÓN DE SEGUNDO ORDEN NO HOMOGÉNEA ---")

    t0 = 0
    T  = 3
    h  = 0.01
    steps = int((T - t0)/h)

    y0 = [2.0, 1.0]  # y(0)=2, y'(0)=1

    ts2, ys2 = rk2_heun(f_segundo_orden, t0, y0, h, steps)
    ts4, ys4 = rk4(f_segundo_orden, t0, y0, h, steps)


    y_exact = y_segundo_orden(ts4, y0)

    error_RK2 = np.max(np.abs(ys2[:,0] - y_exact))
    error_RK4 = np.max(np.abs(ys4[:,0] - y_exact))

    print("Error RK2 =", error_RK2)
    print("Error RK4 =", error_RK4)


    return ts2, ys2, ts4, ys4

def graficar_segundo_orden(ts2, ys2, ts4, ys4):
    """Genera la gráfica de comparación para la ED de Segundo Orden."""
    y0 = [2.0, 1.0] # Usar las C.I. de la prueba
    y_exact = y_segundo_orden(ts4, y0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(ts4, y_exact, 'k--', label="Analítica", linewidth=1.5)
    plt.plot(ts4, ys4[:,0], label="RK4", linewidth=2)
    plt.plot(ts2, ys2[:,0], label="RK2 (Heun)", linestyle=':', linewidth=2)
    
    plt.legend()
    plt.title(r'ED de Segundo Orden: $y^{\prime\prime} + 4y = t^2$')
    plt.xlabel('Tiempo $t$')
    plt.ylabel('Solución $y(t)$')
    plt.grid(True)
    plt.show()


# ===================
# SISTEMA LINEAL 2x2
# u' = A u + f(t)
# A = [[2, 1],
#     [0, 2]]
# f(t) = [   t   ]
#        [ e^{-t}]
# ===================

def f_sistema_lineal(t, u):
    x, y = u
    dx = 2*x + 1*y + t
    dy = 2*y + np.exp(-t)
    return np.array([dx, dy])

def y_sistema_lineal(t):
    exp_neg_t = np.exp(-t)
    exp_2t = np.exp(2*t)
    C1 = 4/3
    C2 = 5/36
    u1 = (1/9)*exp_neg_t + C1*t*exp_2t - 0.5*t - 0.25 + C2*exp_2t
    u2 = - (1/3)*exp_neg_t + C1*exp_2t
    return np.array([u1, u2])

def exacta_sistema_lineal(t):
    t = np.array(t)   # aseguramos que funciona con listas
    exp_neg_t = np.exp(-t)
    exp_2t = np.exp(2*t)

    C1 = 4/3
    C2 = 5/36

    u1 = (1/9)*exp_neg_t + C1*t*exp_2t - 0.5*t - 0.25 + C2*exp_2t
    u2 = - (1/3)*exp_neg_t + C1*exp_2t

    # MUY IMPORTANTE: regresar en forma (N, 2)
    return np.column_stack((u1, u2))


def sistema_lineal():
    print("\n--- SISTEMA LINEAL 2x2 ---")

    t0 = 0
    T  = 3
    h  = 0.01
    steps = int((T - t0)/h)

    u0 = [0.0, 1.0]

    ts2, us2 = rk2_heun(f_sistema_lineal, t0, u0, h, steps)
    ts4, us4 = rk4(f_sistema_lineal, t0, u0, h, steps)

    # Solución analítica evaluada
    u_exact = np.array([y_sistema_lineal(t) for t in ts4])

    error_RK2 = np.max(np.abs(us2 - u_exact))
    error_RK4 = np.max(np.abs(us4 - u_exact))

    print("Error RK2 =", error_RK2)
    print("Error RK4 =", error_RK4)


    return ts2, us2, ts4, us4

def graficar_sistema_lineal(ts2, us2, ts4, us4):
    """Genera las gráficas de comparación para el Sistema Lineal 2x2."""
    u_exact = exacta_sistema_lineal(ts4)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    # FIX: Se simplifica la notación matricial para evitar el error de Matplotlib MathText.
    # Se usa una notación lineal más simple para la matriz A.
    fig.suptitle(r'Sistema Lineal $2 \times 2$: $\mathbf{u}^\prime = A \mathbf{u} + \mathbf{f}(t)$, con $A=[[2, 1], [0, 2]]$', fontsize=14)
    
    # --- Gráfica para u1(t) ---
    axes[0].plot(ts4, u_exact[:,0], 'k--', label="Analítica", linewidth=1.5)
    axes[0].plot(ts4, us4[:,0], label="RK4", linewidth=2)
    axes[0].plot(ts2, us2[:,0], label="RK2 (Heun)", linestyle=':', linewidth=2)
    axes[0].set_title(r'Solución $u_1(t)$ (Variable $x$)')
    axes[0].set_xlabel('Tiempo $t$')
    axes[0].set_ylabel('$u_1(t)$')
    axes[0].legend()
    axes[0].grid(True)

    # --- Gráfica para u2(t) ---
    axes[1].plot(ts4, u_exact[:,1], 'k--', label="Analítica", linewidth=1.5)
    axes[1].plot(ts4, us4[:,1], label="RK4", linewidth=2)
    axes[1].plot(ts2, us2[:,1], label="RK2 (Heun)", linestyle=':', linewidth=2)
    axes[1].set_title(r'Solución $u_2(t)$ (Variable $y$)')
    axes[1].set_xlabel('Tiempo $t$')
    axes[1].set_ylabel('$u_2(t)$')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Ajusta el espacio para el supertítulo
    plt.show()


# =====================================
# SISTEMA NO LINEAL (PÉNDULO FORZADO)
# theta' = omega
# omega' = -(g/L) sin(theta) - k*omega + A*cos(Omega*t)
# =====================================

g = 9.81
L = 1.0
k = 0.5
A = 1.2
Omega = 2.0

def f_no_lineal(t, y):
    theta, omega = y
    return np.array([
        omega,
        - (g/L)*np.sin(theta) - k*omega + A*np.cos(Omega * t)
    ])

def no_lineal():
    print("\n--- SISTEMA NO LINEAL (PÉNDULO FORZADO) ---")

    t0 = 0
    T  = 10
    h  = 0.01
    steps = int((T - t0)/h)

    theta0 = 0.5
    omega0 = 0.0
    y0 = [theta0, omega0]

    ts2, ys2 = rk2_heun(f_no_lineal, t0, y0, h, steps)
    ts4, ys4 = rk4(f_no_lineal, t0, y0, h, steps)

    diff_theta = np.max(np.abs(ys4[:,0] - ys2[:,0]))
    diff_omega = np.max(np.abs(ys4[:,1] - ys2[:,1]))
    print(f"Diferencia máxima en theta RK2 vs RK4 = {diff_theta}")
    print(f"Diferencia máxima en omega RK2 vs RK4 = {diff_omega}")

    return ts2, ys2, ts4, ys4

# =========================
# ESTUDIO DE CONVERGENCIA
# =========================
def convergencia(f, y_exacta_func, t0, T, y0, metodo):
    hs = [0.5, 0.25, 0.125, 0.0625, 0.03125]
    errores = []

    for h in hs:
        steps = int((T - t0) / h)
        if steps < 1:
            steps = 1
            h = T - t0

        ts, ys = metodo(f, t0, y0, h, steps)

        # Evaluar solución analítica
        y_ex = y_exacta_func(ts)

        # --- CASO 1: y_ex devuelve un escalar por cada t ---
        if np.ndim(y_ex) == 1:
            # Si ys es vector de estado (n,2), tomamos solo y(t)
            if ys.ndim == 2:
                error = np.max(np.abs(ys[:,0] - y_ex))
            else:
                error = np.max(np.abs(ys - y_ex))

        # --- CASO 2: y_ex devuelve vector (para sistemas) ---
        else:
            # Error vectorial
            error = np.max(np.abs(ys - y_ex))

        errores.append(error)

    return hs, errores

def plot_convergencia(hs_rk2, errors_rk2, hs_rk4, errors_rk4, title):
    """Grafica el error máximo en función del tamaño de paso (h) en escala log-log."""
    h_rk2, err_rk2 = np.array(hs_rk2), np.array(errors_rk2)
    h_rk4, err_rk4 = np.array(hs_rk4), np.array(errors_rk4)

    plt.figure(figsize=(10, 6))
    
    # --- RK2: Graficar datos y línea de referencia O(h^2) ---
    plt.loglog(h_rk2, err_rk2, 'bo-', label='RK2 (Heun) - Errores')
    p_rk2 = 2.0
    C_rk2 = err_rk2[0] / (h_rk2[0]**p_rk2)
    plt.loglog(h_rk2, C_rk2 * (h_rk2**p_rk2), 'b--', alpha=0.6, label=f'Referencia $O(h^{p_rk2:.0f})$')

    # --- RK4: Graficar datos y línea de referencia O(h^4) ---
    plt.loglog(h_rk4, err_rk4, 'rs-', label='RK4 - Errores')
    p_rk4 = 4.0
    C_rk4 = err_rk4[0] / (h_rk4[0]**p_rk4)
    plt.loglog(h_rk4, C_rk4 * (h_rk4**p_rk4), 'r--', alpha=0.6, label=f'Referencia $O(h^{p_rk4:.0f})$')

    # --- Formato ---
    plt.title(f'Estudio de Convergencia (Escala Log-Log): {title}', fontsize=14)
    plt.xlabel('Tamaño de Paso $h$')
    plt.ylabel('Error Máximo $\\text{max}|y_{\\text{num}} - y_{\\text{ex}}| $')
    plt.legend(loc='lower right')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.gca().invert_xaxis() # Es común invertir el eje x para que h decrezca hacia la derecha
    plt.show()



# ===========
# EJECUCIÓN
# ===========

def main():
    # 1. ED de Primer Orden
    ts2_1, ys2_1, ts4_1, ys4_1 = primer_orden()
    
    # 2. ED de Segundo Orden
    ts2_2, ys2_2, ts4_2, ys4_2 = segundo_orden()
    
    # 3. Sistema Lineal 2x2
    ts2_3, us2_3, ts4_3, us4_3 = sistema_lineal()

    # 4. Sistema No Lineal
    ts2_4, ys2_4, ts4_4, ys4_4 = no_lineal()

# =========================
# ESTUDIO DE CONVERGENCIA
# =========================
    print("\n==============================")
    print("    ESTUDIO DE CONVERGENCIA")
    print("==============================")

    # PRIMER ORDEN
    print("\n--- Convergencia: Primer Orden ---")
    hs, err2 = convergencia(f_primer_orden, y_primer_orden, 0, 3, [1.0], rk2_heun)
    hs, err4 = convergencia(f_primer_orden, y_primer_orden, 0, 3, [1.0], rk4)
    for i in range(len(hs)):
        print(f"h={hs[i]:.5f}   RK2={err2[i]:.8e}   RK4={err4[i]:.8e}")

    # SEGUNDO ORDEN
    print("\n--- Convergencia: Segundo Orden ---")
    hs, err2 = convergencia(f_segundo_orden, lambda t: y_segundo_orden(t, [2.0, 1.0]), 0, 3, [2.0, 1.0], rk2_heun)
    hs, err4 = convergencia(f_segundo_orden,lambda t: y_segundo_orden(t, [2.0, 1.0]),0, 3, [2.0, 1.0], rk4)
    for i in range(len(hs)):
        print(f"h={hs[i]:.5f}   RK2={err2[i]:.8e}   RK4={err4[i]:.8e}")

   # SISTEMA LINEAL 2x2
    print("\n--- Convergencia: Sistema Lineal 2×2 ---")
    hs, err2 = convergencia(f_sistema_lineal, exacta_sistema_lineal, 0, 3, [0.0, 1.0], rk2_heun)
    hs, err4 = convergencia(f_sistema_lineal, exacta_sistema_lineal, 0, 3, [0.0, 1.0], rk4)
    for i in range(len(hs)):
        print(f"h={hs[i]:.5f}   RK2={err2[i]:.8e}   RK4={err4[i]:.8e}")

    plot_convergencia(hs, err2, hs, err4, "ED de Primer Orden")
    plot_convergencia(hs, err2, hs, err4, "ED de Segundo Orden")
    plot_convergencia(hs, err2, hs, err4, "Sistema Lineal 2x2") 

    # ==================
    # GRÁFICAS
    # ==================
    graficar_primer_orden(ts2_1, ys2_1, ts4_1, ys4_1)
    graficar_segundo_orden(ts2_2, ys2_2, ts4_2, ys4_2)
    graficar_sistema_lineal(ts2_3, us2_3, ts4_3, us4_3)
    



if __name__ == "__main__":
    main()
