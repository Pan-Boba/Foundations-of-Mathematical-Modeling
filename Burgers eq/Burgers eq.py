import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# x-y-t grid
nx, ny, t_max = 50, 50, 1
x_min, x_max, y_min, y_max = 0, 3, 0, 3
x, y = np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny)
dx, dy = (x_max - x_min) / nx, (y_max - y_min) / ny
X, Y = np.meshgrid(x, y)

# simulation parameters
CFL, nu, amplitude = 0.1, 0.01, 1.0
if nu == 0:
    dt = CFL * dx * dy
else:
    dt = CFL * dx * dy / nu
nt = int(t_max/dt)
u, v = np.zeros((ny, nx)), np.zeros((ny, nx))

# initial conditions
# hump function
u, v = amplitude * np.exp(- 10 * ((X - 1)**2 + (Y - 1)**2)), amplitude * np.exp(- 10 * ((X - 1)**2 + (Y - 1)**2))

# ladder function
#u, v = amplitude * (X < 1.5), amplitude * (X < 1.5)

# main
z = []
for n in range(nt + 1):
    un, vn = u.copy(), v.copy()

    # variables for animation
    z.append(np.sqrt(un**2 + vn**2))

    u[1:-1, 1:-1] -= dt * (un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) / dx +
                           vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) / dy -
                           nu * ((un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) / dx**2 +
                                 (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]) / dy**2))

    v[1:-1, 1:-1] -= dt * (un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) / dx +
                           vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) / dy -
                           nu * ((vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) / dx**2 +
                                 (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]) / dy**2))

    # periodic boundary conditions for hump function
    u[0, :],  u[-1, :], u[:, 0], u[:, -1] = 0, 0, 0, 0
    v[0, :],  v[-1, :], v[:, 0], v[:, -1] = 0, 0, 0, 0

    # boundary conditions for ladder function
    #u[0, :], u[-1, :], u[:, 0], u[:, -1] = amplitude * (x < 1.5), amplitude * (x < 1.5), amplitude, 0
    #v[0, :], v[-1, :], v[:, 0], v[:, -1] = amplitude * (x < 1.5), amplitude * (x < 1.5), amplitude, 0


# animation
fig = plt.figure(figsize=(11, 8))
plt.suptitle(r'Burgers equation: $\frac{\partial \overrightarrow{u}}{\partial t} + \overrightarrow{u} \, \overrightarrow{\nabla} \, \overrightarrow{u} = \nu \, \nabla^2 \, \overrightarrow{u}$  for $\nu = $' + f'{nu}', fontsize=16)
ax = fig.gca(projection='3d')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlim(0, np.sqrt(2) * amplitude)

time_text = ax.text(x_max, y_max, np.sqrt(2) * amplitude, f't = {t_max}', bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, fontsize=14)
ax.plot_surface(X, Y, z[0], cmap="plasma", rstride=1, cstride=1)


def change_plot(i):
   ax.collections.clear()
   time_text.set_text(f't = {round(i * dt, 4)}')
   ax.plot_surface(X, Y, z[i], cmap="plasma", rstride=1, cstride=1)
   return fig, time_text,


ani = animation.FuncAnimation(fig, change_plot, nt, interval=0, blit=False)
plt.show()


