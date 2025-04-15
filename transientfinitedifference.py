"""
Nathan Cammarata
ENGR 375
Crank-Nicolson Finite Difference Code
----------------------------------------------------------
This script solves the transient 1D heat conduction using
Crank-Nicolson finite difference method
with one of the following tip boundary conditions:
  - 'specified_flux': Adiabatic tip 
  - 'active': Convective tip with surrounding temperature and convection coefficient
  - 'specified_temperature': Specified temperature at the fin tip

Set parameters and tip condition, then run to find:
  - Tip temperature
  - Fin heat transfer rate per unit width
  - Fin efficiency
  - Fin effectiveness
  - Fin thermal resistance
  - Temperature distribution plot
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
k = 180             # W/m·K
h = 100             # W/m²·K
Tb = 373.15         # K 
To_inf = 298.15     # K 
L = 0.01            # m
t = 0.001           # m
P = 2 * t           # Perimeter
Ac = t * 1.0        # Unit width cross-sectional area
rho = 2700          # kg/m^3 
cp = 900            # J/kg·K
alpha = k / (rho * cp)

tip_type = "active"  # 'specified_flux', 'active', or 'specified_temperature'
def tip_temperature(t):
    return 320 + 5 * np.sin(0.001 * t) 

N = 101
M = 10000
x = np.linspace(0, L, N)
dx = L / (N - 1)
dt = 0.01
Fo = alpha * dt / dx**2

T = np.ones((N, M)) * To_inf
T[0, :] = Tb

A = np.zeros((N, N))
B = np.zeros((N, N))
for i in range(1, N - 1):
    A[i, i - 1] = -Fo
    A[i, i]     = 1 + 2 * Fo
    A[i, i + 1] = -Fo

    B[i, i - 1] = Fo
    B[i, i]     = 1 - 2 * Fo
    B[i, i + 1] = Fo

A[0, 0] = B[0, 0] = 1
A[-1, -1] = B[-1, -1] = 1

for n in range(1, M):
    T_prev = T[:, n - 1].copy()
    RHS = B @ T_prev

    if tip_type == "specified_flux":
        A[-1, -2] = -1
        A[-1, -1] = 1
        B[-1, -2] = -1
        B[-1, -1] = 1
        RHS[-1] = 0

    elif tip_type == "active":
        Bi = h * dx / k
        A[-1, -2] = -1
        A[-1, -1] = 1 + Bi
        B[-1, -2] = 1
        B[-1, -1] = -(1 - Bi)
        RHS[-1] = 2 * Bi * To_inf

    elif tip_type == "specified_temperature":
        T_tip = tip_temperature(n * dt)
        A[-1, :] = 0
        A[-1, -1] = 1
        B[-1, :] = 0
        B[-1, -1] = 1
        RHS[-1] = T_tip

    T[:, n] = np.linalg.solve(A, RHS)

T_final = T[:, -1]
dTdx_base = (T_final[1] - T_final[0]) / dx
qf = -k * Ac * dTdx_base
m = np.sqrt(h * P / (k * Ac))
eta_f = qf / (h * Ac * (Tb - To_inf))
eff_f = qf / (h * L * Ac * (Tb - To_inf))
Rf = (Tb - To_inf) / qf

print(f"Tip condition: {tip_type}")
print(f"Tip temperature T(L) = {T_final[-1]:.4f} K")
print(f"Heat transfer rate qf = {qf:.4f} W/m")
print(f"Fin efficiency ηf = {eta_f:.4f}")
print(f"Fin effectiveness εf = {eff_f:.4f}")
print(f"Thermal resistance Rf = {Rf:.4f} K·m/W")

plt.figure(figsize=(6, 5))
time_indices = np.linspace(0, min(M, 500), 10, dtype=int)  

for i in time_indices:
    plt.plot(x, T[:, i], label=f't={i*dt:.2f}s')

plt.xlabel("Distance (x)")
plt.ylabel("Temperature (K)")
plt.title("Transient Temperature Change")
plt.grid(True)
plt.tight_layout()
plt.show()

