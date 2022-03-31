import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# simulation parameters
N = 128
boxsize = 1
dx = boxsize / N
vol = dx ** 2
useSlopeLimiting = False

# 2D spacial grid
xlin = np.linspace(0, boxsize, N)
ylin = np.linspace(0, 2 * boxsize, 2 * N)
Y, X = np.meshgrid(ylin, xlin)

# time interval
courant_fac = 0.5
t, tEnd = 0, 1

# const and directions for np.roll()
gamma, g = 5/3, 0.00001
rho1, rho2 = 1, 0.5
R, L = -1, 1

# initial conditions
rho = rho1 - (rho1-rho2) * (abs(Y - 1.5) < 0.1) * (abs(X - 0.5) < 0.15)
# rho = rho1 - (rho1-rho2) * (Y + 0.1 * np.cos(2 * np.pi * X) > 1)
vx = np.zeros(X.shape)
vy = 1 * np.sin(8 * np.pi * (X - 0.3)) * (np.exp(- 1000 * (Y - 1.4)**2)) * (abs(X - 0.5) < 0.15)
# vy = 1 * (np.sin(np.pi * X))**2 * (np.exp(- 100 * (Y + 0.1 * np.cos(2 * np.pi * X) - 1)**2))
P_int = 2.5 * np.ones(X.shape)


def getConserved(rho, vx, vy, P_int):
    Mass = rho * vol
    Momx, Momy = rho * vx * vol,  rho * vy * vol
    Energy = (P_int/(gamma - 1) + 0.5 * rho * (vx ** 2 + vy ** 2) - rho * g * Y) * vol

    return Mass, Momx, Momy, Energy


def getPrimitive(Mass, Momx, Momy, Energy):
    rho = Mass / vol
    vx, vy = Momx / (rho * vol), Momy / (rho * vol)
    P_int = (Energy / vol - 0.5 * rho * (vx ** 2 + vy ** 2) + rho * g * Y) * (gamma - 1)

    return rho, vx, vy, P_int


def getGradient(f):
    f_dx = (np.roll(f, R, axis=0) - np.roll(f, L, axis=0)) / (2 * dx)
    f_dy = (np.roll(f, R, axis=1) - np.roll(f, L, axis=1)) / (2 * dx)

    return f_dx, f_dy


def slopeLimit(f, dx, f_dx, f_dy):
    f_dx = np.maximum(0., np.minimum(1., ((f - np.roll(f, L, axis=0)) / dx) / (f_dx + 1.0e-8 * (f_dx == 0)))) * f_dx
    f_dx = np.maximum(0., np.minimum(1., (-(f - np.roll(f, R, axis=0)) / dx) / (f_dx + 1.0e-8 * (f_dx == 0)))) * f_dx
    f_dy = np.maximum(0., np.minimum(1., ((f - np.roll(f, L, axis=1)) / dx) / (f_dy + 1.0e-8 * (f_dy == 0)))) * f_dy
    f_dy = np.maximum(0., np.minimum(1., (-(f - np.roll(f, R, axis=1)) / dx) / (f_dy + 1.0e-8 * (f_dy == 0)))) * f_dy

    return f_dx, f_dy


def extrapolateInSpaceToFace(f, f_dx, f_dy):
    f_XL, f_XR = f - f_dx * dx / 2, f + f_dx * dx / 2
    f_XL = np.roll(f_XL, R, axis=0)

    f_YL, f_YR = f - f_dy * dx / 2, f + f_dy * dx / 2
    f_YL = np.roll(f_YL, R, axis=1)

    return f_XL, f_XR, f_YL, f_YR


def getFlux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_int_L, P_int_R):
    # left and right energies
    en_L = P_int_L / (gamma - 1) + 0.5 * rho_L * (vx_L ** 2 + vy_L ** 2) - rho_L * g * (Y - dx)
    en_R = P_int_R / (gamma - 1) + 0.5 * rho_R * (vx_R ** 2 + vy_R ** 2) - rho_R * g * (Y + dx)

    # compute star (averaged) states
    rho_star = 0.5 * (rho_L + rho_R)
    momx_star, momy_star = 0.5 * (rho_L * vx_L + rho_R * vx_R),  0.5 * (rho_L * vy_L + rho_R * vy_R)
    en_star = 0.5 * (en_L + en_R)
    P_int_star = (en_star - 0.5 * (momx_star ** 2 + momy_star ** 2) / rho_star + rho_star * g * Y) * (gamma - 1)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass = momx_star
    flux_Momx, flux_Momy = momx_star ** 2 / rho_star + P_int_star, momx_star * momy_star / rho_star
    flux_Energy = (en_star + P_int_star) * momx_star / rho_star

    # find wavespeeds
    C = np.maximum(np.sqrt(gamma * P_int_L/rho_L) + np.abs(vx_L), np.sqrt(gamma * P_int_R/rho_R) + np.abs(vx_R))

    # add stabilizing diffusive term
    flux_Mass -= C * 0.5 * (rho_L - rho_R)
    flux_Momx -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
    flux_Momy -= C * 0.5 * (rho_L * vy_L - rho_R * vy_R)
    flux_Energy -= C * 0.5 * (en_L - en_R)

    return flux_Mass, flux_Momx, flux_Momy, flux_Energy


def applyFluxes(F, flux_F_X, flux_F_Y, n):
    # update solution
    Flux = dt * dx * (np.roll(flux_F_X, L, axis=0) + np.roll(flux_F_Y, L, axis=1) - flux_F_X - flux_F_Y)
    # boundary condition
    for i in range(N):
        Flux[i][0],  Flux[i][2 * N - 1] = 0, 0
    F += Flux
    if n == 1:
        F -= dt * g * rho

    return F


# main
Mass, Momx, Momy, Energy = getConserved(rho, vx, vy, P_int)
rho_arr, v_arr, i = [], [], 0
while t < tEnd:
    # get Primitive variables
    rho, vx, vy, P_int = getPrimitive(Mass, Momx, Momy, Energy)

    # variables for animation
    rho_arr.append([])
    v_arr.append([])
    rho_arr[i], v_arr[i] = rho, np.sqrt(vx**2 + vy**2)

    # get time step (CFL) = dx / max signal speed
    dt = courant_fac * np.min(dx / (np.sqrt(gamma * P_int / rho) + np.sqrt(vx ** 2 + vy ** 2)))

    # calculate gradients
    rho_dx, rho_dy = getGradient(rho)
    vx_dx, vx_dy = getGradient(vx)
    vy_dx, vy_dy = getGradient(vy)
    P_int_dx, P_int_dy = getGradient(P_int)

    # slope limit gradients
    if useSlopeLimiting:
        rho_dx, rho_dy = slopeLimit(rho, dx, rho_dx, rho_dy)
        vx_dx, vx_dy = slopeLimit(vx, dx, vx_dx, vx_dy)
        vy_dx, vy_dy = slopeLimit(vy, dx, vy_dx, vy_dy)
        P_dx, P_dy = slopeLimit(P_int, dx, P_int_dx, P_int_dy)

    # extrapolate half-step in time
    rho_prime = rho - 0.5 * dt * (vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy)
    vx_prime = vx - 0.5 * dt * (vx * vx_dx + vy * vx_dy + (1 / rho) * P_int_dx)
    vy_prime = vy - 0.5 * dt * (vx * vy_dx + vy * vy_dy + (1 / rho) * P_int_dy + g)
    P_int_prime = P_int - 0.5 * dt * (gamma * P_int * (vx_dx + vy_dy) + vx * P_int_dx + vy * P_int_dy)

    # extrapolate in space to face centers
    rho_XL, rho_XR, rho_YL, rho_YR = extrapolateInSpaceToFace(rho_prime, rho_dx, rho_dy)
    vx_XL, vx_XR, vx_YL, vx_YR = extrapolateInSpaceToFace(vx_prime, vx_dx, vx_dy)
    vy_XL, vy_XR, vy_YL, vy_YR = extrapolateInSpaceToFace(vy_prime, vy_dx, vy_dy)
    P_int_XL, P_int_XR, P_int_YL, P_int_YR = extrapolateInSpaceToFace(P_int_prime, P_int_dx, P_int_dy)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X = getFlux(rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, P_int_XL, P_int_XR)
    flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y = getFlux(rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, P_int_YL, P_int_YR)

    # update solution
    Mass = applyFluxes(Mass, flux_Mass_X, flux_Mass_Y, 0)
    Momx = applyFluxes(Momx, flux_Momx_X, flux_Momx_Y, 0)
    Momy = applyFluxes(Momy, flux_Momy_X, flux_Momy_Y, 1)
    Energy = applyFluxes(Energy, flux_Energy_X, flux_Energy_Y, 0)
    # update time
    t, i = t + dt, i + 1


# animation
fig, axes = plt.subplots(1, 2, figsize=(8, 5))
fig.suptitle(r'R–T instability with ${\rho}_1$ / ${\rho}_2$' + f' = {rho1}/{rho2}. Time interval = {tEnd}', fontsize=16)
fig.subplots_adjust(hspace=0.5, wspace=1)
axes[0].get_xaxis().set_visible(False)
axes[0].get_yaxis().set_visible(True)
axes[0].set_title(r'$\rho(x, y)$', fontsize=16, pad=10)
axes[0].set_ylabel('y', fontsize=14, labelpad=10)

axes[1].get_xaxis().set_visible(False)
axes[1].get_yaxis().set_visible(False)
axes[1].set_title(r'$\overline{v} (x, y)$', fontsize=16, pad=10)

# declaring objects for animation
im_rho = axes[0].imshow(rho_arr[0].T, cmap='viridis', origin="upper", vmin=rho2, vmax=rho1)
plt.colorbar(im_rho, ax=axes[0])
im_v = axes[1].imshow(v_arr[0].T, cmap='plasma')
plt.colorbar(im_v, ax=axes[1])


def animate_func(i):
    im_rho.set_array(rho_arr[i].T)
    im_v.set_array(v_arr[i].T)
    return [im_rho, im_v]


animation.FuncAnimation(fig, animate_func, interval=20, blit=True)

plt.show()
