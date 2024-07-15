import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return 3 * x[0]**2 + 2 * x[0] * x[1] + x[1]**2 + 2 * x[0] + 4 * x[1]

def grad(x):
    df_dx = 6 * x[0] + 2 * x[1] + 2
    df_dy = 2 * x[0] + 2 * x[1] + 4
    return np.array([df_dx, df_dy])

def gradient_descent(x0, y0, learning_rate=0.1, tolerance=1e-6, max_iterations=1000):
    x = np.array([x0, y0])
    func_values = [func(x)]
    coords = [x]

    for i in range(int(1e3)):
        gradient = grad(x)
        new_x = x - learning_rate * gradient

        func_values.append(func(new_x))
        coords.append(new_x)

        if np.linalg.norm(new_x - x) < tolerance:
            print("AAAAAAA")
            break

        x = new_x

    return np.array(coords), np.array(func_values)

def adam_descent(x0, y0, learning_rate=0.1, tolerance=1e-6, max_iterations=1000):
    x = np.array([x0, y0])
    func_values = [func(x)]
    coords = [x]
    beta1=0.9
    beta2=0.999
    eta = 0.1
    m=0
    v=0
    for i in range(int(1e3)):
        gradient = grad(x)
        g = -gradient
        beta1=beta1
        beta2=beta2
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*g**2
        m_v = m/(1-beta1)
        v_v = v/(1-beta2)

        new_x = x + eta*(m_v)/(np.sqrt(v_v)+1e-8)
        func_values.append(func(new_x))
        coords.append(new_x)

        if np.linalg.norm(new_x - x) < tolerance:
            print("AAAAAAA")
            break

        x = new_x

    return np.array(coords), np.array(func_values)



coords1, func_values1 = gradient_descent(2, 3, 0.1, 1e-3)
coords2, func_values2 = adam_descent(2, 3, 0.1, 1e-3)
plt.figure(figsize=(8,8))
plt.plot(func_values1)
plt.plot(func_values2)
plt.savefig("b", format='pdf')
x = np.linspace(-4, 4, 400)
y = np.linspace(-4, 4, 400)
X, Y = np.meshgrid(x, y)
Z = func([X,Y])

plt.figure(figsize=(10, 6))
plt.contourf(X, Y, Z, 50, cmap='viridis')
plt.colorbar()
plt.plot(coords1[:, 0], coords1[:, 1], 'r.-', markersize=10, label='Gradient Descent Path')
plt.scatter(coords1[0, 0], coords1[0, 1], color='red', label='Start', zorder=5)
plt.scatter(coords1[-1, 0], coords1[-1, 1], color='red', label='End', zorder=5)

plt.plot(coords2[:, 0], coords2[:, 1], 'b.-', markersize=10, label='Adam Descent Path')
plt.scatter(coords2[0, 0], coords2[0, 1], color='blue', label='Start', zorder=5)
plt.scatter(coords2[-1, 0], coords2[-1, 1], color='blue', label='End', zorder=5)
plt.title('Gradient Descent Path on Contour Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.savefig("a", format='pdf')





