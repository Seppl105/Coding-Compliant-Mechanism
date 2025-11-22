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
t41, t12, t22, t32, t42, alpha1= sp.symbols('t41 t12 t22 t32 t42 alpha1', real=True)

# the location of the local coordinate system 2 is unknowen at the start
T2x, T2y, Phi2 = sp.symbols('T2x T2y Phi2', real=True)

# Input variables:
t21 = np.deg2rad(94)

# Knowen variables and relations:
r11, r21, r31, r41 = 1.8, 2.5, 2.0, 2.2
r12, r22, r32, r42 = 1.8, 2.5, 2.0, 2.2
t11 = np.deg2rad(0)
################t12 = np.deg2rad(3)
k1 = 0.2
k2_x, k2_y = -1, 1.5
l_c = 0.2 # length of the connecting link
t31 = t32 + np.deg2rad(15)
T1 = [0, 0] # offset/translation of local coordinate system 1 to global coordinate system
Phi1 = 0 # rotation of local coordinate system 1 with respect to the global coordinate system
l_A1A2 = 5
t_A1A2 = t21 + np.deg2rad(-285)###################
Phi2 = t_A1A2 #local x-axis of system two aligns with the line A1A2



# All vector loop equations in the form R1 + R2 + ... + Rn = 0 for the real and imaginary part respectively
eqs = []
# Local vector loop in system 1
eqs += [ r21*sp.cos(t21) + r31*sp.cos(t31) - (r11*sp.cos(t11) + r41*sp.cos(t41)) ]
eqs += [ r21*sp.sin(t21) + r31*sp.sin(t31) - (r11*sp.sin(t11) + r41*sp.sin(t41)) ] 
# Local vector loop in system 2
eqs += [ r22*sp.cos(t22) + r32*sp.cos(t32) - (r12*sp.cos(t12) + r42*sp.cos(t42)) ]
eqs += [ r22*sp.sin(t22) + r32*sp.sin(t32) - (r12*sp.sin(t12) + r42*sp.sin(t42)) ]

# Endpoint of r21
A1_loc = sp.Matrix([r21*sp.cos(t21), r21*sp.sin(t21)])
# Endpoint of r41
B1_loc = sp.Matrix([r41*sp.cos(t41), r41*sp.sin(t41)])
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


####### add explanation for the conversion
# Vector loop equation incorporating the connecting link
eqs += r21*sp.cos(t21) + l_A1A2*sp.cos(t_A1A2) - r22*sp.cos(t22 + Phi2) + ( k2_x*sp.cos(Phi2) - k2_y*sp.sin(Phi2) ) - l_c*sp.cos(alpha1) - r41*(1+k1)*sp.cos(t41) # note that [k2_x, k2_y] was converted using relations knowen from polar coordinates 
eqs += r21*sp.sin(t21) + l_A1A2*sp.sin(t_A1A2) - r22*sp.sin(t22 + Phi2) + ( k2_x*sp.sin(Phi2) + k2_y*sp.cos(Phi2) ) - l_c*sp.sin(alpha1) - r41*(1+k1)*sp.sin(t41) # note that [k2_x, k2_y] was converted using relations knowen from polar coordinates



