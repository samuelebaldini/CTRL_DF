import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Estensione della funzione al dominio 3D
def func(x):
    return 2 * x[0]**2 + 2 * x[0] * x[1] + x[1]**2 + 2 * x[0] + 4 * x[1] + 5 * x[2]**2 + x[2]

# Estensione del gradiente al dominio 3D
def grad(x):
    df_dx = 4 * x[0] + 2 * x[1] + 2
    df_dy = 2 * x[0] + 2 * x[1] + 4
    df_dz = 10 * x[2] + 1
    return np.array([df_dx, df_dy, df_dz])

def gradient_descent(x0, y0, z0, learning_rate=10, tolerance=1e-6, max_iterations=1000):
    x = np.array([x0, y0, z0])
    func_values = [func(x)]
    coords = [x]

    for i in range(max_iterations):
        gradient = grad(x)
        new_x = x - learning_rate * gradient

        func_values.append(func(new_x))
        coords.append(new_x)

        if np.linalg.norm(new_x - x) < tolerance:
            break

        x = new_x

    return np.array(coords), np.array(func_values),i

def adam_descent(x0, y0, z0, learning_rate=100, tolerance=1e-6, max_iterations=1000):
    x = np.array([x0, y0, z0])
    func_values = [func(x)]
    coords = [x]
    beta1 = 0.9
    beta2 = 0.999
    eta = learning_rate
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for i in range(max_iterations):
        gradient = grad(x)
        g = gradient
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_v = m / (1 - beta1)
        v_v = v / (1 - beta2)

        new_x = x - eta * m_v / (np.sqrt(v_v) + 1e-8)
        func_values.append(func(new_x))
        coords.append(new_x)

        if np.linalg.norm(new_x - x) < tolerance:
            break

        x = new_x

    return np.array(coords), np.array(func_values),i

def conjugate_gradient(x0, learning_rate=0.01, tolerance=1e-6, max_iterations=1000):
    x = np.array(x0)
    func_values = [func(x)]
    coords = [x]

    g_old = grad(x)
    s = -g_old

    for i in range(max_iterations):
        new_x = x + learning_rate * s
        func_values.append(func(new_x))
        coords.append(new_x)

        g_new = grad(new_x)
        beta = np.dot(g_new, g_new) / np.dot(g_old, g_old)
        s = -g_new + beta * s

        if np.linalg.norm(new_x - x) < tolerance:
            break

        x = new_x
        g_old = g_new

    return np.array(coords), np.array(func_values),i


# Esecuzione della discesa del gradiente e dell'adam
coords1, func_values1, it1 = gradient_descent(2, 3, 1.5, 0.1, 1e-3)
coords2, func_values2, it2 = adam_descent(2, 3, 1.5, 0.25, 1e-3)
coords3, func_values3, it3 = conjugate_gradient([2, 3, 1.5], 0.1, 1e-3)
# Creazione della figura 3D con Matplotlib
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Traiettoria Gradient Descent
ax.plot(coords1[:, 0], coords1[:, 1], coords1[:, 2], 'r.-', markersize=4, label=f'Gradient Descent Path: {it1}')

# Traiettoria Adam Descent
ax.plot(coords2[:, 0], coords2[:, 1], coords2[:, 2], 'b.-', markersize=4, label=f'Adam Descent Path: {it2}')

ax.plot(coords3[:, 0], coords3[:, 1], coords3[:, 2], 'g.-', markersize=4, label=f'Conjugate Gradient Path: {it3}')
# Configurazione del layout e della prospettiva

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()

# Modifica della prospettiva
ax.view_init(elev=15, azim=+230)  # Cambia questi valori per ottenere la prospettiva desiderata

# Salva l'immagine come file PDF di alta qualità
plt.savefig("traiettorie_3d_alta_qualita.pdf", format='pdf', dpi=300)
plt.show()
