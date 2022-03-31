import matplotlib.pyplot as plt
import numpy as np


# const and x-grid
n, x_min, x_max, iter_number = 500, -10, 10, 10
h = (x_max - x_min)/n
x_data = np.linspace(x_min, x_max, n)


# potential field
def U(x):
    return pow(x, 2)/2


# bound state problem
a = [0] + [-1/(2 * pow(h, 2)) for _ in range(1, n)]
b = [1/pow(h, 2) + U(x) for x in x_data]
c = [-1/(2 * pow(h, 2)) for _ in range(0, n - 1)] + [0]


def solution(x):
    # for U(x) = x^2/2
    return np.exp(- pow(x, 2)/2)/pow(np.pi, 0.25)


def tridiagonal_matrix_algorithm(d):
    s, t = c.copy(), d.copy()
    for i in range(0, n):
        s[i] = -s[i]/(b[i] + a[i] * s[i-1])
        t[i] = -(a[i] * t[i-1] - t[i])/(b[i] + a[i] * s[i-1])

    y = t.copy()
    for i in range(n - 2, -1, -1):
        y[i] = s[i] * y[i+1] + y[i]
    return y


def inverse_iteration_method(y):
    eigen_vector_i = y.copy()
    for k in range(0, iter_number - 2):
        eigen_vector_i = tridiagonal_matrix_algorithm(eigen_vector_i)
    eigen_vector = tridiagonal_matrix_algorithm(eigen_vector_i)

    # defining norm of vector as max
    eigen_val = max(eigen_vector_i)/max(eigen_vector)

    # normalised wave-function
    eigen_vector = eigen_vector/(max(eigen_vector) * pow(np.pi, 0.25))
    return eigen_val, eigen_vector


E, psi = inverse_iteration_method(np.array([0.5 for x in range(n)]))
print('Calculated energy of ground state is ', E)
print('Real energy of ground state is 0.5')


# graphics
fig, axes = plt.subplots(2, 1, figsize=(8, 8))
axes[0].set_xlim(x_min, x_max)
axes[0].set_title(r'Normalised wave-function $\psi(x)$ of the Schrödinger equation in $U(x) = \frac{x^2}{2}$')
axes[0].scatter(x_data, solution(x_data), color='black', label='real solution', s=10)
axes[0].plot(x_data, psi, color='darkorange', label='inverse iteration')
axes[0].text(-9, 0.65, f'n = {n} \nNumber of iterations = {iter_number}', bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5})
axes[0].text(4, 0.5, r'$U(x) = \pi^{-1/4} \cdot \exp{\left(-\frac{x^2}{2}\right)}$', color='black', fontsize=12)
axes[0].legend(loc='upper right')

axes[1].set_xlim(x_min, x_max)
axes[1].set_title('Error value')
axes[1].plot(x_data, solution(x_data) - psi, c='red')

plt.show()
