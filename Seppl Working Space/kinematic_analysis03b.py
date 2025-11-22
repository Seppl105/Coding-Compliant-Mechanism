import numpy as np
import sympy as sp
import itertools

# --------------------------------------------
# Symbols (keep your style)
# --------------------------------------------
t41, t12, t22, t32, t42, alpha1 = sp.symbols('t41 t12 t22 t32 t42 alpha1', real=True)
t31 = sp.Symbol('t31', real=True)  # decoupled for system-1 solve, then tied: t31 = t32 + 15°

# --------------------------------------------
# Inputs / geometry (your values)
# --------------------------------------------
t21 = np.deg2rad(95)

# System 1
r11, r21, r31_val, r41 = 1.9, 2.8, 2.1, 2.4
t11 = np.deg2rad(0)

# System 2 (lengths)
r12, r22_val, r32_val, r42 = 1.9, 2.9, 2.0, 3.0

# Connector & placements
k1 = 0.54
k2_x, k2_y = -1.8, 2.9
l_c = 1.4
T1 = [0, 0]
Phi1 = 0
l_A1A2 = 5.4
t_A1A2 = t21 + np.deg2rad(-285)    # = t21 - 285°
Phi2 = t_A1A2                      # local x-axis of system 2 aligns with A1A2

# --------------------------------------------
# (1) System 1 only: solve for t41, t31
#     r21 e^(it21) + r31 e^(it31) = r11 e^(it11) + r41 e^(it41)
# --------------------------------------------
eq1 = r21*sp.cos(t21) + r31_val*sp.cos(t31) - (r11*sp.cos(t11) + r41*sp.cos(t41))
eq2 = r21*sp.sin(t21) + r31_val*sp.sin(t31) - (r11*sp.sin(t11) + r41*sp.sin(t41))

# Try a couple of seeds for robustness
seeds_sys1 = [
    (np.deg2rad(20),  np.deg2rad(10)),
    (np.deg2rad(-30), np.deg2rad(25)),
    (np.deg2rad(40),  np.deg2rad(-15)),
]
sol_sys1 = None
for s41, s31 in seeds_sys1:
    try:
        sol = sp.nsolve((eq1, eq2), (t41, t31), (s41, s31))
        sol_sys1 = (float(sol[0]), float(sol[1]))
        break
    except Exception:
        pass
if sol_sys1 is None:
    raise RuntimeError("System 1 failed to solve — try different seeds.")

t41_s1, t31_s1 = sol_sys1
print("[System 1] t41 = %7.3f deg | t31 = %7.3f deg" %
      (np.rad2deg(t41_s1), np.rad2deg(t31_s1)))
print("[System 1] residuals:",
      float(eq1.subs({t41:t41_s1, t31:t31_s1})),
      float(eq2.subs({t41:t41_s1, t31:t31_s1})))

# --------------------------------------------
# (2) Full system (6 equations, 6 unknowns)
#     Tie t31 to t32 with the given 15° offset
# --------------------------------------------
t31_expr = t32 + np.deg2rad(15)

# Local vector loop in system 1 (reuse but with t31=t32+15°)
eq1_full = r21*sp.cos(t21) + r31_val*sp.cos(t31_expr) - (r11*sp.cos(t11) + r41*sp.cos(t41))
eq2_full = r21*sp.sin(t21) + r31_val*sp.sin(t31_expr) - (r11*sp.sin(t11) + r41*sp.sin(t41))

# Local vector loop in system 2
eq3 = r22_val*sp.cos(t22) + r32_val*sp.cos(t32) - (r12*sp.cos(t12) + r42*sp.cos(t42))
eq4 = r22_val*sp.sin(t22) + r32_val*sp.sin(t32) - (r12*sp.sin(t12) + r42*sp.sin(t42))

# Connecting link equations (your form)
eq5 = ( r21*sp.cos(t21) + l_A1A2*sp.cos(t_A1A2)
        - r22_val*sp.cos(t22 + Phi2)
        + ( k2_x*sp.cos(Phi2) - k2_y*sp.sin(Phi2) )
        - l_c*sp.cos(alpha1)
        - r41*(1+k1)*sp.cos(t41) )
eq6 = ( r21*sp.sin(t21) + l_A1A2*sp.sin(t_A1A2)
        - r22_val*sp.sin(t22 + Phi2)
        + ( k2_x*sp.sin(Phi2) + k2_y*sp.cos(Phi2) )
        - l_c*sp.sin(alpha1)
        - r41*(1+k1)*sp.sin(t41) )

unknowns = (t41, t12, t22, t32, t42, alpha1)

# --------------------------------------------
# Seeds: treat t32 carefully
# Start from t31_s1 and back out t32 ≈ t31 − 15°
# Then perturb t32 across a small set to catch the right branch.
# --------------------------------------------
t32_base = t31_s1 - np.deg2rad(15)

t32_candidates = [t32_base + np.deg2rad(d) for d in (-40, -20, -10, -5, 0, 5, 10, 20, 40)]
t41_candidates = [t41_s1 + np.deg2rad(d) for d in (-20, 0, 20)]
t12_candidates = [np.deg2rad(d) for d in (-30, 0, 30)]
t22_candidates = [np.deg2rad(d) for d in (120, 150, 180)]
t42_candidates = [np.deg2rad(d) for d in (20, 40, 60)]
a1_candidates  = [np.deg2rad(d) for d in (0, 20, 40)]

best = None
for g41, g12, g22, g32, g42, ga1 in itertools.product(
        t41_candidates, t12_candidates, t22_candidates, t32_candidates, t42_candidates, a1_candidates):
    guess = (g41, g12, g22, g32, g42, ga1)
    try:
        sol = sp.nsolve((eq1_full, eq2_full, eq3, eq4, eq5, eq6), unknowns, guess, tol=1e-14, maxsteps=100)
        vals = tuple(float(s) for s in sol)
        # score by residual norm
        r1 = float(eq1_full.subs(dict(zip(unknowns, vals))))
        r2 = float(eq2_full.subs(dict(zip(unknowns, vals))))
        r3 = float(eq3.subs(dict(zip(unknowns, vals))))
        r4 = float(eq4.subs(dict(zip(unknowns, vals))))
        r5 = float(eq5.subs(dict(zip(unknowns, vals))))
        r6 = float(eq6.subs(dict(zip(unknowns, vals))))
        res = np.sqrt(r1*r1 + r2*r2 + r3*r3 + r4*r4 + r5*r5 + r6*r6)
        if (best is None) or (res < best[0]):
            best = (res, vals)
        # early exit if very tight
        if res < 1e-9:
            break
    except Exception:
        continue

if best is None:
    raise RuntimeError("Full system failed to solve — broaden the seed ranges a bit.")

resnorm, (t41_v, t12_v, t22_v, t32_v, t42_v, a1_v) = best
t31_v = t32_v + np.deg2rad(15)

print("\n[Full system] solution (degrees)")
print("t41 = %8.3f" % np.rad2deg(t41_v))
print("t12 = %8.3f" % np.rad2deg(t12_v))
print("t22 = %8.3f" % np.rad2deg(t22_v))
print("t32 = %8.3f" % np.rad2deg(t32_v))
print("t31 = %8.3f  (t32 + 15°)" % np.rad2deg(t31_v))
print("t42 = %8.3f" % np.rad2deg(t42_v))
print("alpha1 = %8.3f" % np.rad2deg(a1_v))
print("[Full system] residual norm: %.3e" % resnorm)

# Individual residuals (useful for debugging)
for i, e in enumerate([eq1_full, eq2_full, eq3, eq4, eq5, eq6], start=1):
    print(f"  eq{i}: {float(e.subs({t41:t41_v, t12:t12_v, t22:t22_v, t32:t32_v, t42:t42_v, alpha1:a1_v})):.3e}")

# --------------------------------------------
# (Optional) System 1 plotting — stays clean
# --------------------------------------------
# Ground pivots
O2_1 = sp.Matrix([0.0, 0.0])
O4_1 = sp.Matrix([r11*sp.cos(t11), r11*sp.sin(t11)])

# Endpoints of input and output
A1_loc = sp.Matrix([r21*sp.cos(t21), r21*sp.sin(t21)])          # O2_1 -> A1
B1_loc = O4_1 + sp.Matrix([r41*sp.cos(t41), r41*sp.sin(t41)])   # O4_1 -> B1

# Coupler is A1 -> B1
A1_n = np.array(A1_loc, dtype=float).flatten()
B1_n = np.array(B1_loc.subs({t41:t41_v}), dtype=float).flatten()

# Sanity: |B1-A1| ≈ r31
AB = B1_n - A1_n
print("\n[Check] |B1 - A1| = %.6f (target r31 = %.6f)" % (np.linalg.norm(AB), r31_val))
