# Four-bar linkage proof of concept: solve loop-closure with SymPy and plot one configuration
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# -----------------------------
# Parameters (edit these)
# Ground-pivot distance (O2->O4) and link lengths
r1 = 100.0  # ground
r2 = 40.0   # input (crank)
r3 = 120.0  # coupler
r4 = 80.0   # output (rocker)

# Input angle in radians (0 = along +x)
theta2 = np.deg2rad(50.0)

# Initial guesses for unknown angles (radians). Choose near expected configuration.
theta3_guess = np.deg2rad(120.0)
theta4_guess = np.deg2rad(70.0)
# -----------------------------

# Unknowns
t3, t4 = sp.symbols('t3 t4', real=True)

# Vector-loop equations (split into x/y components)
eqx = sp.Eq(r2*sp.cos(theta2) + r3*sp.cos(t3), r1 + r4*sp.cos(t4))
eqy = sp.Eq(r2*sp.sin(theta2) + r3*sp.sin(t3), r4*sp.sin(t4))

# Solve numerically
sol = sp.nsolve([eqx.lhs - eqx.rhs, eqy.lhs - eqy.rhs],
                [t3, t4],
                [theta3_guess, theta4_guess])
theta3_val = float(sol[0])
theta4_val = float(sol[1])

# Joint coordinates
O2 = np.array([0.0, 0.0])
O4 = np.array([r1, 0.0])
A = O2 + np.array([r2*np.cos(theta2), r2*np.sin(theta2)])
B = O4 + np.array([r4*np.cos(theta4_val), r4*np.sin(theta4_val)])

# Residual check of closure
closure_residual = np.linalg.norm(A + np.array([r3*np.cos(theta3_val), r3*np.sin(theta3_val)]) - B)

# Plot one configuration
fig, ax = plt.subplots(figsize=(6,3))
# Links: ground, input, coupler, output
ax.plot([O2[0], O4[0]], [O2[1], O4[1]], '-o', label='ground')
ax.plot([O2[0], A[0]], [O2[1], A[1]], '-o', label='input')
ax.plot([A[0], B[0]], [A[1], B[1]], '-o', label='coupler')
ax.plot([O4[0], B[0]], [O4[1], B[1]], '-o', label='output')
ax.axis('equal')
ax.set_title('Four-bar linkage (single configuration)')
ax.legend(loc='upper right')
ax.grid(True)

theta2_deg = np.rad2deg(theta2)
theta3_deg = np.rad2deg(theta3_val)
theta4_deg = np.rad2deg(theta4_val)

print(f"Solved angles (deg): θ2={theta2_deg:.3f}, θ3={theta3_deg:.3f}, θ4={theta4_deg:.3f}")
print(f"Closure residual (should be ~0): {closure_residual:.3e}")

plt.show()