import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Modeling one upper and one lower segment by subdividing it into two coupled four bar linkages
# named system 1 and system 2 respectively

# Conventions:
# Local variables (v) are given with indices (ij) corresponding to the link (i) and the system (j)
#   e.g.: v_32 is local variable v corresponding to link 3 and system 2
# tij is short for theta_ij measured CCW from the x-axis
# rij represents length ij
# O2j and O4j are the local ground points in system j of segments 2 and 4 respectively
# Aj and Bj are the local endpoints of r2j and r4j respectively

# Rotation matrix (probably familiar from polar coordinate systems)
# Used to transform a local coordinate system rotated CCW by th back into the global coordinate system
# def R(phi):
#     return sp.Matrix([[sp.cos(phi), -sp.sin(phi)],
#                       [sp.sin(phi),  sp.cos(phi)]])
def R(phi):
    return np.array([[np.cos(phi), -np.sin(phi)],
                      [np.sin(phi),  np.cos(phi)]])

# Unknowen variables and relations:
# unknowen angles theta and the angles of the connecting joint
t31, t41, t12, t22, t42, alpha1= sp.symbols('t31 t41 t12 t22 t42 alpha1', real=True)

### # the location of the local coordinate system 2 is unknowen at the start
# T2x, T2y, Phi2 = sp.symbols('T2x T2y Phi2', real=True)

# Input variables:
t21 = np.deg2rad(95)

### Knowen variables and relations:
# system 1 lengths
r11, r21, r31, r41 = 1.9,2.8,2.1,2.4#1.8, 2.5, 2.0, 2.2
# system 2 lengths
r12, r22, r32, r42 = 1.9, 2.9, 2, 3#1.8, 2.5, 2.0, 2.2
t11 = np.deg2rad(0)
################t12 = np.deg2rad(3)
k1 = 0.54
k2_x, k2_y = -1.8, 2.9
l_c = 2.71#1.4 # length of the connecting link
t32 = t31 - np.deg2rad(15)
T1 = [0, 0] # offset/translation of local coordinate system 1 to global coordinate system
Phi1 = 0 # rotation of local coordinate system 1 with respect to the global coordinate system
l_A1A2 = 5.4
t_A1A2 = t21 + np.deg2rad(-285)###################
Phi2 = t_A1A2 #local x-axis of system two aligns with the line A1A2



# All vector loop equations in the form R1 + R2 + ... + Rn = 0 for the real and imaginary part respectively for sys1
eqs_sys1 = []
# Local vector loop in system 1
# r21*e^(i t21) + r31*e^(i t31) = r11*e^(i t11) + r41*e^(i t41)
eqs_sys1.append( r21*sp.cos(t21) + r31*sp.cos(t31) - (r11*sp.cos(t11) + r41*sp.cos(t41)) )
eqs_sys1.append( r21*sp.sin(t21) + r31*sp.sin(t31) - (r11*sp.sin(t11) + r41*sp.sin(t41)) )

# Solve for t41, t31
x0_sys1 = [np.deg2rad(20.0),  # t41 initial
           np.deg2rad(10.0)]  # t31 initial
unknowns_sys1 = [t41, t31]
sol_sys1 = sp.nsolve(eqs_sys1, unknowns_sys1, x0_sys1)
t41_val = float(sol_sys1[0])
t31_val = float(sol_sys1[1])
t32_val = float(t31_val - np.deg2rad(15.0))
result_sys1 = {t41: t41_val, t31: t31_val}

eqs_sys2 = []
# Local vector loop in system 2
eqs_sys2.append( r22*sp.cos(t22) + r32*sp.cos(t32) - (r12*sp.cos(t12) + r42*sp.cos(t42)) )
eqs_sys2.append( r22*sp.sin(t22) + r32*sp.sin(t32) - (r12*sp.sin(t12) + r42*sp.sin(t42)) )
####### add explanation for the conversion
# Vector loop equation incorporating the connecting link
eqs_sys2.append( r21*sp.cos(t21) + l_A1A2*sp.cos(t_A1A2) - r22*sp.cos(t22 + Phi2) + ( k2_x*sp.cos(Phi2) - k2_y*sp.sin(Phi2) ) - l_c*sp.cos(alpha1) - r41*(1+k1)*sp.cos(t41) ) # note that [k2_x, k2_y] was converted using relations knowen from polar coordinates 
eqs_sys2.append( r21*sp.sin(t21) + l_A1A2*sp.sin(t_A1A2) - r22*sp.sin(t22 + Phi2) + ( k2_x*sp.sin(Phi2) + k2_y*sp.cos(Phi2) ) - l_c*sp.sin(alpha1) - r41*(1+k1)*sp.sin(t41) ) # note that [k2_x, k2_y] was converted using relations knowen from polar coordinates

eqs_sys2 = [e.subs(result_sys1) for e in eqs_sys2]


x0_sys2 = [
    np.deg2rad(0.0),    # t12
    np.deg2rad(90.0),  # t22
    np.deg2rad(90.0),   # t42
    np.deg2rad(-20.0),   # alpha1
]
unknowns_sys2 = [t12, t22, t42, alpha1]
sol_sys2 = sp.nsolve(eqs_sys2, unknowns_sys2, x0_sys2, tol=1e-14, maxsteps=100)
t12_val, t22_val, t42_val, alpha1_val = map(float, sol_sys2)

################# Plotting

# Startpoint of r21
O2_1_loc = sp.Matrix([0.0, 0.0])
# Endpoint of r21
A1_loc = sp.Matrix([r21*sp.cos(t21), r21*sp.sin(t21)])
# Startpont of r41
O4_1_loc = sp.Matrix([r11*sp.cos(t11), r11*sp.sin(t11)])
# Endpoint of r41
B1_loc = O4_1_loc + sp.Matrix([r41*sp.cos(t41), r41*sp.sin(t41)])
# Fixation point for connecting linkage to system 1
P1_loc = B1_loc + sp.Matrix([k1*r41*sp.cos(t41), k1*r41*sp.sin(t41)]) ### here specified as an extension of r41 by factor k1

# Startpoint of r22
O2_2_loc = sp.Matrix([0,0])
# Endpoint of r22
A2_loc = sp.Matrix([r22*sp.cos(t22), r22*sp.sin(t22)])
# Startpoint of r42
O4_2_loc = sp.Matrix([r12*sp.cos(t12), r12*sp.sin(t12)])
# Endpoint of r42
B2_loc = sp.Matrix([r42*sp.cos(t42), r42*sp.sin(t42)])
# Fixation point for connecting linkage to system 2
P2_loc = sp.Matrix([k2_x, k2_y]) ### here specified as an offset to O2 by k2_x and k2_y



### Evaluate Positions

#subs_sys1 = {t41: t41_val, t31: t31_val}
# O21_n = np.array(O2_1_loc, dtype=float).flatten()
# O41_n = np.array(O4_1_loc, dtype=float).flatten()
# A1_n   = np.array(A1_loc, dtype=float).flatten()
# B1_n   = np.array(B1_loc.subs(subs_sys1), dtype=float).flatten()

# to numpy helper
to_np = lambda m: np.array(m.evalf(subs=subs_all), dtype=float).flatten()

subs_all = {
    t41: t41_val, t31: t31_val,
    t12: t12_val, t22: t22_val, t42: t42_val, alpha1: alpha1_val
}

# System 1 already in global form
O2_1_global = to_np(O2_1_loc)
A1_global   = to_np(A1_loc)
O4_1_global = to_np(O4_1_loc)
B1_global   = to_np(B1_loc)
P1_global   = to_np(P1_loc)

# System 2
##########O2_2_global = A1_global + l_A1A2 * np.array([np.cos(t_A1A2), np.sin(t_A1A2)], dtype=float)
A2_global = A1_global + l_A1A2 * np.array([np.cos(t_A1A2), np.sin(t_A1A2)], dtype=float)
# We know for every point P holds P =  O2_2_global + R2 @ P_local (translation and rotation)
# Plug in A2 and rearange to get:
############A2_global = O2_2_global + R(Phi2) @ to_np(A2_loc)
O2_2_global = A2_global - R(Phi2) @ to_np(A2_loc) 
# And use the expression for P to evaluate the remaining global coordinates
O4_2_global = O2_2_global + R(Phi2) @ to_np(O4_2_loc)
B2_global   = O2_2_global + R(Phi2) @ to_np( (O4_2_loc + B2_loc))
P2_global   = O2_2_global + R(Phi2) @ to_np(P2_loc)


### Sanity checks
print("alpha1: ", alpha1_val)
print("t12 t22 t32 t42: ", np.rad2deg(t12_val), np.rad2deg(t22_val), np.rad2deg(t32_val), np.rad2deg(t42_val))
conn_len = np.linalg.norm(P2_global - P1_global)
print(f"[Check] |P2 - P1| = {conn_len:.6f} (target l_c = {l_c:.6f})")
print(f"[Check] |B1 - A1| = {np.linalg.norm(B1_global - A1_global):.6f} (target r31 = {r31:.6f})")
print(f"[Check] |B2 - A2| = {np.linalg.norm(B2_global - A2_global):.6f} (target r32 = {r32:.6f})")

########################
# sanity check
plt.figure()
plt.plot([to_np(O2_2_loc[0]), to_np(O4_2_loc[0])], [to_np(O2_2_loc[1]), to_np(O4_2_loc[1])],
         '-', linewidth=2, label='r12')
plt.plot([to_np(O2_2_loc[0]), to_np(A2_loc[0])],   [to_np(O2_2_loc[1]), to_np(A2_loc[1])],
         '-', label='r22')
plt.plot([to_np(O4_2_loc[0]), to_np(B2_loc[0])],   [to_np(O4_2_loc[1]), to_np(B2_loc[1])],
         '-', label='r42')
plt.plot([to_np(A2_loc[0]),   to_np(B2_loc[0])],   [to_np(A2_loc[1]),   to_np(B2_loc[1])],
         '--', label='r32')
plt.legend()
plt.show()
###################################

plt.figure()
plt.title("Both four-bars")

# --- System 1 ---
plt.plot([O2_1_global[0], O4_1_global[0]], [O2_1_global[1], O4_1_global[1]],
         '-', linewidth=2, label='r11 (ground)')
plt.plot([O2_1_global[0], A1_global[0]],   [O2_1_global[1], A1_global[1]],
         '-', label='r21')
plt.plot([O4_1_global[0], B1_global[0]],   [O4_1_global[1], B1_global[1]],
         '-', label='r41')
plt.plot([A1_global[0],   B1_global[0]],   [A1_global[1],   B1_global[1]],
         '--', label='r31')

# --- System 2 ---
# r12 ground: O2_2 -> O4_2
plt.plot([O2_2_global[0], O4_2_global[0]], [O2_2_global[1], O4_2_global[1]],
         '-', linewidth=2, label='r12 (ground)')
# r22: O2_2 -> A2
plt.plot([O2_2_global[0], A2_global[0]],   [O2_2_global[1], A2_global[1]],
         '-', label='r22')
# r42: O4_2 -> B2
plt.plot([O4_2_global[0], B2_global[0]],   [O4_2_global[1], B2_global[1]],
         '-', label='r42')
# r32 coupler: A2 -> B2
plt.plot([A2_global[0],   B2_global[0]],   [A2_global[1],   B2_global[1]],
         '--', label='r32')

# --- Connector ---
plt.plot([P1_global[0], P2_global[0]], [P1_global[1], P2_global[1]],
         '-.', label='connector l_c')
plt.scatter([P1_global[0], P2_global[0]], [P1_global[1], P2_global[1]], s=20)

# Joints (small markers)
for pt in (O2_1_global, O4_1_global, A1_global, B1_global,
           O2_2_global, O4_2_global, A2_global, B2_global):
    plt.plot(pt[0], pt[1], 'o', ms=4)

plt.axis('equal')
plt.xlabel('x'); plt.ylabel('y')
plt.legend(loc='best')
plt.show()

# # Sanity checks for the coupler (A1 -> B1)
# AB = B1_n - A1_n
# AB_len = np.linalg.norm(AB)
# AB_ang = np.arctan2(AB[1], AB[0])
# print(f"|B1 - A1| = {AB_len:.6f} (should be r31 = {r31:.6f})")
# ang_err = np.rad2deg((AB_ang - t31_val + np.pi) % (2*np.pi) - np.pi)  # error in [-180,180)
# print(f"angle(B1 - A1) - t31 = {ang_err:.6f} deg")


# plt.figure()
# plt.title("Sketch")

# # ground link O2_1 -> O4_1
# plt.plot([O21_n[0], O41_n[0]], [O21_n[1], O41_n[1]], '-', linewidth=2, label='ground r11')

# # input O2_1 -> A1
# plt.plot([O21_n[0], A1_n[0]], [O21_n[1], A1_n[1]], '-', label='r21')

# # output O4_1 -> B1
# plt.plot([O41_n[0], B1_n[0]], [O41_n[1], B1_n[1]], '-', label='r41')

# # coupler A1 -> B1  (THIS is the correct coupler segment)
# plt.plot([A1_n[0], B1_n[0]], [A1_n[1], B1_n[1]], '--', label='r31 (coupler A1→B1)')

# # joints
# plt.plot(*O21_n, 'o'); plt.plot(*O41_n, 'o'); plt.plot(*A1_n, 'o'); plt.plot(*B1_n, 'o')

# plt.axis('equal'); plt.xlabel('x'); plt.ylabel('y'); plt.legend()
# plt.show()
