import numpy as np
import matplotlib.pyplot as plt
#########################################################################################################################
"""
Steady-State Finite Difference Computational Mini Project
Nathan Cammarata
March 24, 2025

Instructions:
Specify grid resolution by changing N on line 77.
Specify spacing between blades/fins on line 80
Specify boundary conditions and thermal properties from lines 88-92.

"""
def apply_boundary_conditions(h, Node, k, ho, To_inf, hi, Ti_inf, dx, q_flux=0):
    """
    Applies boundary conditions, including convection, insulation, and heat flux.
    """
    Nx, Ny = np.shape(h)
    
    for j in range(Ny):
        for i in range(Nx):
            if Node[i, j] == 1:
                continue  # Fixed Dirichlet Boundary Condition
            
            # Safe indexing to prevent out-of-bounds errors
            im1 = max(i - 1, 0)
            ip1 = min(i + 1, Nx - 1)
            jm1 = max(j - 1, 0)
            jp1 = min(j + 1, Ny - 1)
            
            if Node[i, j] == 21:
                h[i, j] = h[ip1, j]  # No-flux left boundary (Neumann)
            elif Node[i, j] == 22:
                h[i, j] = h[im1, j]  # No-flux right boundary
            elif Node[i, j] == 23:
                h[i, j] = h[i, jp1]  # No-flux bottom boundary
            elif Node[i, j] == 24:
                h[i, j] = h[i, jm1]  # No-flux top boundary

            # Plane surface with convection (Case 3)
            elif Node[i, j] == 32:
                h[i, j] = (2 * ho * dx * To_inf + (h[ip1, j] + h[im1, j] + h[i, jp1] + h[i, jm1])) / (2 * ho * dx + 4)
            
            # Internal corner with convection (Case 2)
            elif Node[i, j] == 33:
                h[i, j] = (2 * (h[ip1, j] + h[im1, j] + h[i, jp1] + h[i, jm1]) + 
                           (2 * ho * dx * To_inf) - ((ho * dx / k) * 3 * h[i, j])) / (4 + 2 * ho * dx / k)
            
            # External corner with convection (Case 4)
            elif Node[i, j] == 34:
                h[i, j] = ((h[im1, j] + h[i, jp1]) + 2 * ho * dx * To_inf - 2 * ((ho * dx / k) + 1) * h[i, j]) / (2 + 2 * ho * dx / k)
            
            # Plane surface with uniform heat flux (Case 5)
            elif Node[i, j] == 35:
                h[i, j] = (2 * (h[im1, j] + h[i, jp1]) + (2 * q_flux * dx / k) - 4 * h[i, j]) / 4

    return h

def gauss_seidel_solver(h, Node, tol=1e-5, max_iters=10000):
    """Solves the system using Gauss-Seidel with boundary conditions."""
    Nx, Ny = np.shape(h)
    lamb = 1.5  # Over-relaxation factor
    max_dif = tol + 1
    iters = 0
    while max_dif > tol and iters < max_iters:
        iters += 1
        max_dif = 0
        for j in range(1, Ny-1):
            for i in range(1, Nx-1):
                if Node[i, j] == 0:  # Internal node
                    h_old = h[i, j]
                    h_new = (h[i+1, j] + h[i-1, j] + h[i, j+1] + h[i, j-1]) / 4
                    h_new = lamb * h_new + (1 - lamb) * h_old
                    h[i, j] = h_new
                    max_dif = max(max_dif, abs(h_new - h_old))
    return h

# Set grid resolution (N)
N = 25 
Nx, Ny = N, N
dx = 6 / (N - 1)  

# Initialize arrays
h = np.zeros((Nx, Ny))
Node = np.zeros((Nx, Ny))

# Thermal properties and boundary conditions
k = 25       # Thermal conductivity
ho = 1000    # Convection coefficient (outside)
To_inf = 1700  # Ambient temperature outside
hi = 200     # Convection coefficient (inside channel)
Ti_inf = 400  # Coolant temperature inside

# Set boundary conditions
h[:, -1] = To_inf  # Right boundary
h[:, 0] = Ti_inf   # Left boundary
Node[:, -1] = 1  # Dirichlet (fixed) boundary 
Node[:, 0] = 1  # Dirichlet (fixed) boundary 

# Apply initial guess
h[1:-1, 1:-1] = (To_inf + Ti_inf) / 2
h = apply_boundary_conditions(h, Node, k, ho, To_inf, hi, Ti_inf, dx)

# Solve the system
h = gauss_seidel_solver(h, Node)

# Plot results
plt.imshow(h.T, origin='lower', cmap='hot', extent=[0, 6, 0, 6])
plt.colorbar(label='Temperature (K)')
plt.title("Temperature Distribution")
plt.xlabel("Width (mm)")
plt.ylabel("Height (mm)")
plt.show()
