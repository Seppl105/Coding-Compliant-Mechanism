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

# # Rotation matrix (probably familiar from polar coordinate systems)
# # Used to transform a local coordinate system rotated CCW by th back into the global coordinate system
# def R(phi):
#     return sp.Matrix([[sp.cos(phi), -sp.sin(phi)],
#                       [sp.sin(phi),  sp.cos(phi)]])
    
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
l_c = 1.4 # length of the connecting link
t32 = t31 - np.deg2rad(15)
T1 = [0, 0] # offset/translation of local coordinate system 1 to global coordinate system
Phi1 = 0 # rotation of local coordinate system 1 with respect to the global coordinate system
l_A1A2 = 5.4
t_A1A2 = t21 + np.deg2rad(-285)###################
Phi2 = t_A1A2 #local x-axis of system two aligns with the line A1A2



# All vector loop equations in the form R1 + R2 + ... + Rn = 0 for the real and imaginary part respectively
eqs = []
# Local vector loop in system 1
# r21*e^(i t21) + r31*e^(i t31) = r11*e^(i t11) + r41*e^(i t41)
eqs.append( r21*sp.cos(t21) + r31*sp.cos(t31) - (r11*sp.cos(t11) + r41*sp.cos(t41)) )
eqs.append( r21*sp.sin(t21) + r31*sp.sin(t31) - (r11*sp.sin(t11) + r41*sp.sin(t41)) )
# # Local vector loop in system 2
# eqs.append( r22*sp.cos(t22) + r32*sp.cos(t32) - (r12*sp.cos(t12) + r42*sp.cos(t42)) )
# eqs.append( r22*sp.sin(t22) + r32*sp.sin(t32) - (r12*sp.sin(t12) + r42*sp.sin(t42)) )

# ####### add explanation for the conversion
# # Vector loop equation incorporating the connecting link
# eqs.append( r21*sp.cos(t21) + l_A1A2*sp.cos(t_A1A2) - r22*sp.cos(t22 + Phi2) + ( k2_x*sp.cos(Phi2) - k2_y*sp.sin(Phi2) ) - l_c*sp.cos(alpha1) - r41*(1+k1)*sp.cos(t41) )# note that [k2_x, k2_y] was converted using relations knowen from polar coordinates 
# eqs.append( r21*sp.sin(t21) + l_A1A2*sp.sin(t_A1A2) - r22*sp.sin(t22 + Phi2) + ( k2_x*sp.sin(Phi2) + k2_y*sp.cos(Phi2) ) - l_c*sp.sin(alpha1) - r41*(1+k1)*sp.sin(t41) )# note that [k2_x, k2_y] was converted using relations knowen from polar coordinates

# Solve for t41, t31
x0_sys1 = [np.deg2rad(20.0),  # t41 initial
           np.deg2rad(10.0)]  # t31 initial
sol_sys1 = sp.nsolve(eqs, [t41, t31], x0_sys1)
t41_val = float(sol_sys1[0])
t31_val = float(sol_sys1[1])



################# Plotting

# Startpoint of r21
O2_1 = sp.Matrix([0.0, 0.0])
# Endpoint of r21
A1_loc = sp.Matrix([r21*sp.cos(t21), r21*sp.sin(t21)])
# Startpont of r41
O4_1 = sp.Matrix([r11*sp.cos(t11), r11*sp.sin(t11)])
# Endpoint of r41
B1_loc = O4_1 + sp.Matrix([r41*sp.cos(t41), r41*sp.sin(t41)])
# Fixation point for connecting linkage to system 1
### here specified as an extension of r41 by factor k1
P1_loc = B1_loc + sp.Matrix([k1*r41*sp.cos(t41), k1*r41*sp.sin(t41)]) 

# Endpoint of r22
A2_loc = sp.Matrix([r22*sp.cos(t22), r22*sp.sin(t22)])
# Endpoint of r42
B2_loc = sp.Matrix([r42*sp.cos(t42), r42*sp.sin(t42)])
# Fixation point for connecting linkage to system 2
### here specified as an offset to O2 by k2_x and k2_y
P2_loc = sp.Matrix([k2_x, k2_y])



### Numeric
subs_sys1 = {t41: t41_val, t31: t31_val}
O21_n = np.array(O2_1, dtype=float).flatten()
O41_n = np.array(O4_1, dtype=float).flatten()
A1_n   = np.array(A1_loc, dtype=float).flatten()
B1_n   = np.array(B1_loc.subs(subs_sys1), dtype=float).flatten()


# Sanity checks for the coupler (A1 -> B1)
AB = B1_n - A1_n
AB_len = np.linalg.norm(AB)
AB_ang = np.arctan2(AB[1], AB[0])
print(f"|B1 - A1| = {AB_len:.6f} (should be r31 = {r31:.6f})")
ang_err = np.rad2deg((AB_ang - t31_val + np.pi) % (2*np.pi) - np.pi)  # error in [-180,180)
print(f"angle(B1 - A1) - t31 = {ang_err:.6f} deg")


plt.figure()
plt.title("Sketch")

# ground link O2_1 -> O4_1
plt.plot([O21_n[0], O41_n[0]], [O21_n[1], O41_n[1]], '-', linewidth=2, label='ground r11')

# input O2_1 -> A1
plt.plot([O21_n[0], A1_n[0]], [O21_n[1], A1_n[1]], '-', label='r21')

# output O4_1 -> B1
plt.plot([O41_n[0], B1_n[0]], [O41_n[1], B1_n[1]], '-', label='r41')

# coupler A1 -> B1  (THIS is the correct coupler segment)
plt.plot([A1_n[0], B1_n[0]], [A1_n[1], B1_n[1]], '--', label='r31 (coupler A1→B1)')

# joints
plt.plot(*O21_n, 'o'); plt.plot(*O41_n, 'o'); plt.plot(*A1_n, 'o'); plt.plot(*B1_n, 'o')

plt.axis('equal'); plt.xlabel('x'); plt.ylabel('y'); plt.legend()
plt.show()

# # Unknown vector
# X = sp.Matrix([t41, t12, t22, t32, t42, alpha1])

# # Convert eq list to matrix
# F = sp.Matrix(eqs)

# # Choose an initial guess (adjust if solver fails)
# guess = sp.Matrix(np.deg2rad([90, -15, 65, -15, 55, 280]))

# sol = sp.nsolve(F, X, guess)
# sol = np.array(sol, dtype=float).ravel()

# t41_v, t12_v, t22_v, t32_v, t42_v, alpha1_v = sol
# print("Solution (radians):", sol)
# print("Solution (degrees):", np.rad2deg(sol))

# def plot_configuration():
#     t31_v = float(t32_v + np.deg2rad(15))

#     # System 1 joints
#     O2_1 = np.array([0,0])
#     O4_1 = np.array([r11*np.cos(t11), r11*np.sin(t11)])
#     A1   = np.array([r21*np.cos(t21), r21*np.sin(t21)])
#     B1   = A1 + np.array([r31*np.cos(t31_v), r31*np.sin(t31_v)])
#     D1   = O4_1 + np.array([r41*np.cos(t41_v), r41*np.sin(t41_v)])
#     P1   = D1 + np.array([k1*r41*np.cos(t41_v), k1*r41*np.sin(t41_v)])

#     # System 2 frame rotation
#     Phi2 = t_A1A2
#     R2 = np.array([[np.cos(Phi2), -np.sin(Phi2)],
#                    [np.sin(Phi2),  np.cos(Phi2)]])

#     A2 = A1 + np.array([l_A1A2*np.cos(Phi2), l_A1A2*np.sin(Phi2)])
#     T2 = A2 - R2 @ np.array([r22*np.cos(t22_v), r22*np.sin(t22_v)])

#     B2 = A2 + R2 @ np.array([r32*np.cos(t32_v), r32*np.sin(t32_v)])
#     D2 = T2 + R2 @ np.array([r42*np.cos(t42_v), r42*np.sin(t42_v)])
#     P2 = T2 + R2 @ np.array([k2_x, k2_y])
#     P2_tip = P2 + l_c*np.array([np.cos(alpha1_v), np.sin(alpha1_v)])

#     fig, ax = plt.subplots()
#     for p,q in [(O2_1,A1),(A1,B1),(O4_1,D1),(B1,D1)]:
#         ax.plot([p[0],q[0]],[p[1],q[1]],'r-')
#     for p,q in [(T2,A2),(A2,B2),(T2,D2),(B2,D2)]:
#         ax.plot([p[0],q[0]],[p[1],q[1]],'b-')

#     ax.plot([P2[0],P2_tip[0]],[P2[1],P2_tip[1]],'g-',lw=2)
#     ax.scatter([P1[0],P2[0]],[P1[1],P2[1]],color='g')

#     ax.axis('equal')
#     ax.grid(True)
#     plt.show()

# plot_configuration()