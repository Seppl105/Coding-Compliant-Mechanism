import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ----------------------------
# Unknowns (system 1 only)
# ----------------------------
t41, t31 = sp.symbols('t41 t31', real=True)
# Keep other symbols declared (unused here) for later integration with system 2
t12, t22, t32, t42, alpha1 = sp.symbols('t12 t22 t32 t42 alpha1', real=True)

# ----------------------------
# Inputs / known geometry
# ----------------------------
t21 = np.deg2rad(95.0)

# system 1 lengths
r11, r21, r31, r41 = 1.9, 2.8, 2.1, 2.4
t11 = np.deg2rad(0.0)

# (system 2 params kept for later; unused here)
r12, r22, r32_val, r42 = 1.9, 2.9, 2.0, 3.0
k1 = 0.54
k2_x, k2_y = -1.8, 2.9
l_c = 1.4
T1 = [0, 0]
Phi1 = 0
l_A1A2 = 5.4
t_A1A2 = t21 + np.deg2rad(-285)
Phi2 = t_A1A2

# ----------------------------
# Vector loop equations (system 1)
# r21*e^(i t21) + r31*e^(i t31) = r11*e^(i t11) + r41*e^(i t41)
# ----------------------------
eqs = []
eqs.append( r21*sp.cos(t21) + r31*sp.cos(t31) - (r11*sp.cos(t11) + r41*sp.cos(t41)) )
eqs.append( r21*sp.sin(t21) + r31*sp.sin(t31) - (r11*sp.sin(t11) + r41*sp.sin(t41)) )

# Solve for t41, t31
x0_sys1 = [np.deg2rad(20.0),  # t41 initial
           np.deg2rad(10.0)]  # t31 initial
sol_sys1 = sp.nsolve(eqs, [t41, t31], x0_sys1)
t41_val = float(sol_sys1[0])
t31_val = float(sol_sys1[1])

print("System 1 solution:")
print(f"  t41 = {np.rad2deg(t41_val):7.3f} deg")
print(f"  t31 = {np.rad2deg(t31_val):7.3f} deg")
print("Residuals:",
      float(eqs[0].subs({t41:t41_val, t31:t31_val})),
      float(eqs[1].subs({t41:t41_val, t31:t31_val})))

# ----------------------------
# Points (system 1, local frame)
# ----------------------------
# Ground pivots
O2_1 = sp.Matrix([0.0, 0.0])
O4_1 = sp.Matrix([r11*sp.cos(t11), r11*sp.sin(t11)])

# Endpoints of input and output links
A1_loc = sp.Matrix([r21*sp.cos(t21), r21*sp.sin(t21)])          # O2_1 -> A1
B1_loc = O4_1 + sp.Matrix([r41*sp.cos(t41), r41*sp.sin(t41)])   # O4_1 -> B1

# Numeric
subs_sys1 = {t41: t41_val, t31: t31_val}
O2_1_n = np.array(O2_1, dtype=float).flatten()
O4_1_n = np.array(O4_1, dtype=float).flatten()
A1_n   = np.array(A1_loc, dtype=float).flatten()
B1_n   = np.array(B1_loc.subs(subs_sys1), dtype=float).flatten()

# ----------------------------
# Sanity checks for the coupler (A1 -> B1)
# ----------------------------
AB = B1_n - A1_n
AB_len = np.linalg.norm(AB)
AB_ang = np.arctan2(AB[1], AB[0])
print(f"|B1 - A1| = {AB_len:.6f} (should be r31 = {r31:.6f})")
ang_err = np.rad2deg((AB_ang - t31_val + np.pi) % (2*np.pi) - np.pi)  # error in [-180,180)
print(f"angle(B1 - A1) - t31 = {ang_err:.6f} deg")

# ----------------------------
# Plot (coupler is A1 -> B1)
# ----------------------------
plt.figure()
plt.title("Four-bar (system 1) — coupler A1→B1")

# ground link O2_1 -> O4_1
plt.plot([O2_1_n[0], O4_1_n[0]], [O2_1_n[1], O4_1_n[1]], '-', linewidth=2, label='ground r11')

# input O2_1 -> A1
plt.plot([O2_1_n[0], A1_n[0]], [O2_1_n[1], A1_n[1]], '-', label='r21')

# output O4_1 -> B1
plt.plot([O4_1_n[0], B1_n[0]], [O4_1_n[1], B1_n[1]], '-', label='r41')

# coupler A1 -> B1  (THIS is the correct coupler segment)
plt.plot([A1_n[0], B1_n[0]], [A1_n[1], B1_n[1]], '--', label='r31 (coupler A1→B1)')

# joints
plt.plot(*O2_1_n, 'o'); plt.plot(*O4_1_n, 'o'); plt.plot(*A1_n, 'o'); plt.plot(*B1_n, 'o')

plt.axis('equal'); plt.xlabel('x'); plt.ylabel('y'); plt.legend()
plt.show()


# import numpy as np
# import sympy as sp
# import matplotlib.pyplot as plt

# # --- symbols (only what we need for system 1) ---
# t41, t31 = sp.symbols('t41 t31', real=True)

# # --- inputs & constants for system 1 ---
# t21 = np.deg2rad(95.0)
# r11, r21, r31, r41 = 1.9, 2.8, 2.1, 2.4
# t11 = 0.0  # ground link along +x from O2 to O4

# # --- system 1 loop: r21 + r31 = r11 + r41  (real/imag) ---
# eq1 = r21*sp.cos(t21) + r31*sp.cos(t31) - (r11*sp.cos(t11) + r41*sp.cos(t41))
# eq2 = r21*sp.sin(t21) + r31*sp.sin(t31) - (r11*sp.sin(t11) + r41*sp.sin(t41))

# # --- solve for [t41, t31] (pick reasonable guesses; change if needed) ---
# x0 = [np.deg2rad(20.0),  # t41
#       np.deg2rad(10.0)]  # t31
# t41_val, t31_val = map(float, sp.nsolve([eq1, eq2], [t41, t31], x0))

# print("t41 = %.2f deg  |  t31 = %.2f deg" % (np.rad2deg(t41_val), np.rad2deg(t31_val)))
# print("closure residuals:",
#       float(eq1.subs({t41:t41_val, t31:t31_val})),
#       float(eq2.subs({t41:t41_val, t31:t31_val})))

# # --- geometry for plotting ---
# O2 = np.array([0.0, 0.0])                  # ground point for link 2
# O4 = np.array([r11*np.cos(t11), r11*np.sin(t11)])  # ground point for link 4

# A  = O2 + np.array([r21*np.cos(t21), r21*np.sin(t21)])   # tip of input crank r21
# B  = O4 + np.array([r41*np.cos(t41_val), r41*np.sin(t41_val)])  # tip of rocker r41
# A_via_O4 = O4 + np.array([r31*np.cos(t31_val), r31*np.sin(t31_val)])  # should coincide with A

# # --- plot: show ground, the two cranks, coupler A–B, and a marker for A via O4 ---
# plt.figure()
# # ground
# plt.plot([O2[0], O4[0]], [O2[1], O4[1]], '-', linewidth=2)
# # input link r21
# plt.plot([O2[0], A[0]], [O2[1], A[1]], '-')
# # output link r41
# plt.plot([O4[0], B[0]], [O4[1], B[1]], '-')
# # coupler r31 + closing A–B
# plt.plot([O4[0], A_via_O4[0]], [O4[1], A_via_O4[1]], '--')
# plt.plot([A[0], B[0]], [A[1], B[1]], '-')

# # joints
# plt.plot(*O2, 'o'); plt.plot(*O4, 'o'); plt.plot(*A, 'o'); plt.plot(*B, 'o')
# # sanity: A computed two ways should coincide
# plt.plot([A[0], A_via_O4[0]], [A[1], A_via_O4[1]], ':')  # should be tiny/overlapped

# plt.axis('equal'); plt.xlabel('x'); plt.ylabel('y')
# plt.title('System 1 four-bar (decoupled from system 2)')
# plt.show()
