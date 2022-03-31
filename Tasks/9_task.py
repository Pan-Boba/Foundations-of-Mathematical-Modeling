import matplotlib.pyplot as plt
import numpy as np


# const
n = 200

# boundary eq: a_0 * y(x_min) + b_0 * y'(x_min) = d_0, a_1 * y(x_max) + b_1 * y'(x_max) = d_1
a_0, b_0, d_0 = [1, 0, 1]
a_1, b_1, d_1 = [1, 0, 1]


def f(x):
    # y'' = f(x) = sin(x), x_min = 0 < x < pi = x_max
    return np.sin(x), 0, np.pi, 'sin x'


# x interval
x_min, x_max = f(0)[1], f(0)[2]
h = (x_max - x_min)/n


# real solution
def solution(x):
    # y = c_2 * x + c_1 - sin(x)
    def boundary(a, b, x_point):
        return a, a * x_point + b

    # detQ != 0
    boundary_1, boundary_2 = boundary(a_0, b_0, x_min), boundary(a_1, b_1, x_max)
    if boundary_1[0] * boundary_2[1] - boundary_1[1] * boundary_2[0] == 0:
        raise Exception('The boundary value problem is unsolvable')

    # A * (c_1, c_2)^T = F
    A = np.array([boundary_1, boundary_2])
    F = np.array([[d_0 + a_0 * np.sin(x_min) + b_0 * np.cos(x_min)], [d_1 + a_1 * np.sin(x_max) + b_1 * np.cos(x_max)]])
    const = np.linalg.solve(A, F)
    c_1, c_2 = const[0], const[1]
    return c_2 * x + c_1 - np.sin(x)


def tridiagonal_matrix_algorithm(x):
    a = [0] + [1/pow(h, 2) for _ in range(1, n - 1)] + [-b_1/h]
    b = [a_0 - b_0/h] + [- 2/pow(h, 2) for _ in range(1, n - 1)] + [a_1 + b_1/h]
    c = [b_0/h] + [1/pow(h, 2) for _ in range(1, n - 1)] + [0]
    g = [d_0] + [f(i)[0] for i in x[1:n-1]] + [d_1]

    s, t = c.copy(), g.copy()
    s[0], t[0] = -s[0]/b[0], t[0]/b[0]
    for i in range(1, n):
        s[i] = -s[i]/(b[i] + a[i] * s[i-1])
        t[i] = -(a[i] * t[i-1] - t[i])/(b[i] + a[i] * s[i-1])

    y = t.copy()
    for i in range(n - 2, -1, -1):
        y[i] = s[i] * y[i+1] + y[i]
    return y


x_data = np.linspace(x_min, x_max, n)
y_alg = tridiagonal_matrix_algorithm(x_data)
y_real = solution(x_data)


# graphics
fig, axes = plt.subplots(2, 1, figsize=(8, 8))
axes[0].set_xlim(x_min, x_max)
axes[0].set_title(r'Tridiagonal matrix algorithm for $y^{\prime\prime}$ = ' + f'{f(0)[3]}')
axes[0].plot(x_data, y_alg, c='darkorange', label='TDMA')
axes[0].plot(x_data, y_real, c='black', label='Solution y(x)', linestyle='dashed')
axes[0].text(2, 0.95, f'n = {n}', bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5})
axes[0].legend(loc='upper right')

axes[1].set_xlim(x_min, x_max)
axes[1].set_title('Error value')
axes[1].plot(x_data, y_real - y_alg, c='red')

plt.show()
