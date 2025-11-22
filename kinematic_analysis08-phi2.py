import math
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# ==== Input data ==========================================================
# four_bar_linkages can be in random order
# coupler_linkes must be in order from left to right and facing from left to right ##############
four_bar_linkages = [
    [
        [57.4614744999, -6.809163619299999, 55.1250449484, 3.6654495336],
        [55.1250449484, 3.6654495336, 45.12855621119999, 3.400472314199999],
        [48.61556714979999, -6.4613681972, 57.4614744999, -6.809163619299999],
        [45.12855621119999, 3.400472314199999, 48.61556714979999, -6.4613681972],
    ],
    [
        [70.8454695603, 10.9262967653, 81.5963284964, 10.9262973727],
        [81.5963284964, 10.9262973727, 80.0, -1.4780103543],
        [70.0, -1.4780103543, 70.8454695603, 10.9262967653],
        [80.0, -1.4780103543, 70.0, -1.4780103543],
    ],
    [
        [131.0698510394, 3.7963544218, 129.4714844591, -0.8429064217],
        [129.4714844591, -0.8429064217, 119.4854498204, -0.3145957228],
        [121.0261760569, 4.046826698, 131.0698510394, 3.7963544218],
        [119.4854498204, -0.3145957228, 121.0261760569, 4.046826698],
    ],
    [
        [94.82478502209997, -2.6784809349, 93.9732988925, 4.825811383199999],
        [94.82478502209997, -2.6784809349, 104.5854619019, -2.9372075712],
        [93.9732988925, 4.825811383199999, 103.946200862, 4.090130538399999],
        [104.5854619019, -2.9372075712, 103.946200862, 4.090130538399999],
    ],
]

coupler_links = [
    [55.8698004946, 9.5132738931, 58.86793206260001, 9.4074102978],
    [85.61870658990001, -3.2093366617, 87.61608151709999, -3.1068993637],
    [105.8473830683, 5.26720582, 108.7778450626, 4.6250278854],
]

# ==== Helper functions ====================================================

def midpoint(line):
    x1, y1, x2, y2 = line
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def length(line):
    x1, y1, x2, y2 = line
    return math.hypot(x2 - x1, y2 - y1)

def angle_deg(line):
    x1, y1, x2, y2 = line
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

def orient_left_to_right(line):
    x1, y1, x2, y2 = line
    if x2 < x1:
        return [x2, y2, x1, y1]
    return line

def orient_bottom_to_top(line):
    x1, y1, x2, y2 = line
    if y2 < y1:
        return [x2, y2, x1, y1]
    return line

def order_links_rectangular(linkage):
    """
    Order links as:
      r1: bottom (ground), left -> right
      r2: left side, bottom -> top
      r3: top, left -> right
      r4: right side, bottom -> top

    Uses midpoints to classify bottom/top and left/right.
    Assumes roughly quadrilateral / four-bar geometry.
    """
    if len(linkage) != 4:
        raise ValueError("Each four-bar linkage must have exactly 4 links.")

    lines = [list(line) for line in linkage]
    mids  = [midpoint(line) for line in lines]

    # Separate bottom vs top using midpoint y
    ys = [m[1] for m in mids]
    min_y = min(ys)
    max_y = max(ys)

    # Thresholds to distinguish bottom vs top
    # (handles negative y naturally)
    tol_y = 0.1 * (max_y - min_y + 1e-9)  # avoid zero range

    bottom_indices = [i for i, (xm, ym) in enumerate(mids) if ym <= min_y + tol_y]
    top_indices    = [i for i, (xm, ym) in enumerate(mids) if ym >= max_y - tol_y]

    # If thresholds were too tight/loose, fall back to simply lowest & highest
    if not bottom_indices:
        bottom_indices = [ys.index(min_y)]
    if not top_indices:
        top_indices = [ys.index(max_y)]

    # Choose r1 (bottom) among bottom_indices: the most horizontal (smallest |angle|)
    def horiz_score(i):
        a = angle_deg(lines[i])
        # consider angles modulo 180 to measure closeness to horizontal
        a_mod = abs(((a + 90) % 180) - 90)
        return a_mod

    r1_idx = min(bottom_indices, key=horiz_score)

    # Choose r3 (top) among top_indices: also most horizontal
    r3_idx = min(top_indices, key=horiz_score)

    # Remaining indices are r2 and r4 (sides)
    remaining = [i for i in range(4) if i not in (r1_idx, r3_idx)]
    if len(remaining) != 2:
        raise RuntimeError("Side link detection failed; check geometry.")

    # r2 = left side (smaller x midpoint), r4 = right side
    x_remaining = [mids[i][0] for i in remaining]
    left_idx = remaining[x_remaining.index(min(x_remaining))]
    right_idx = remaining[x_remaining.index(max(x_remaining))]

    r2_idx, r4_idx = left_idx, right_idx

    # Now orient each according to your rules
    r1 = orient_left_to_right(lines[r1_idx])
    r3 = orient_left_to_right(lines[r3_idx])
    r2 = orient_bottom_to_top(lines[r2_idx])
    r4 = orient_bottom_to_top(lines[r4_idx])

    # Return in order r1, r2, r3, r4
    return [r1, r2, r3, r4]

# ==== Apply ordering + compute properties =================================

ordered_linkages = []

print("=== Ordered four-bar linkages (r1..r4 CCW) ===\n")
for i, linkage in enumerate(four_bar_linkages):
    ordered = order_links_rectangular(linkage)
    ordered_linkages.append(ordered)

    print(f"Mechanism {i+1}:")
    for j, line in enumerate(ordered, start=1):
        name = f"r{j}{i+1}"  # r11, r12, r13, r14, etc. ################ i+1 and j switched
        print(line)
        L = length(line)
        ang = angle_deg(line)
        x1, y1, x2, y2 = line
        print(
            f"  {name}: start=({x1:.3f}, {y1:.3f}), "
            f"end=({x2:.3f}, {y2:.3f}), "
            f"length={L:.3f}, angle={ang:.3f} deg"
        )
    print()

print("Coupler links:")
for j, line in enumerate(coupler_links, start=1):
    name = f"c{j}"
    print(line)
    L = length(line)
    ang = angle_deg(line)
    x1, y1, x2, y2 = line
    print(
            f"  {name}: start=({x1:.3f}, {y1:.3f}), "
            f"end=({x2:.3f}, {y2:.3f}), "
            f"length={L:.3f}, angle={ang:.3f} deg"
        )

# ==== Plotting ============================================================

fig, ax = plt.subplots(figsize=(9, 4))
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

for i, ordered in enumerate(ordered_linkages):
    color = colors[i % len(colors)]
    for j, line in enumerate(ordered, start=1):
        name = f"r{j}{i+1}" ############################# i+1 and j switched
        x1, y1, x2, y2 = line

        # only label the mechanism once in the legend
        lbl = f"Mech {i+1}" if j == 1 else None
        ax.plot([x1, x2], [y1, y2],
                marker="o", linestyle="-", color=color, label=lbl)

        xm, ym = midpoint(line)
        ax.text(xm, ym, name, fontsize=7, ha="center", va="center")

# plot coupler links for reference
for k, line in enumerate(coupler_links):
    x1, y1, x2, y2 = line
    lbl = "Coupler links" if k == 0 else None
    ax.plot([x1, x2], [y1, y2],
            marker="s", linestyle="--", color="black", label=lbl)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Four-bar mechanisms ordered as r1..r4\n"
             "r1 bottom, r2 left, r3 top, r4 right")
ax.set_aspect("equal", adjustable="box")
ax.grid(True, linestyle=":", linewidth=0.5)
#ax.legend()
plt.tight_layout()
plt.show()

# ==== Kinematic analysis ==================================================


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
t21 = np.deg2rad(109.473) #np.deg2rad(95)

# ===== Knowen variables and relations ===================================
### Old:
# # system 1 lengths
# r11, r21, r31, r41 = 1.9,2.8,2.1,2.4#1.8, 2.5, 2.0, 2.2
# # system 2 lengths
# r12, r22, r32, r42 = 1.9, 2.9, 2, 3#1.8, 2.5, 2.0, 2.2
# t11 = np.deg2rad(0)
# ################t12 = np.deg2rad(3)
# k1 = 0.54
# k2_x, k2_y = -1.8, 2.9
# l_c = 2.71#1.4 # length of the connecting link
# t32 = t31 - np.deg2rad(15)
# T1 = [0, 0] # offset/translation of local coordinate system 1 to global coordinate system
# Phi1 = 0 # rotation of local coordinate system 1 with respect to the global coordinate system
# l_A1A2 = 5.4
# t_A1A2 = t21 + np.deg2rad(-285)###################
# Phi2 = t_A1A2 #local x-axis of system two aligns with the line A1A2

### New:
# System 1: take mechanism 1 (index 0)
sys1_links = ordered_linkages[0]   # [r1, r2, r3, r4] for mechanism 1
r11, r21, r31, r41 = [length(line) for line in sys1_links]

# System 2: take mechanism 2 (index 1)
sys2_links = ordered_linkages[1]   # [r1, r2, r3, r4] for mechanism 2
r12, r22, r32, r42 = [length(line) for line in sys2_links]

print("\nLengths from geometry:")
print("System 1 (mech 1): r11, r21, r31, r41 =", r11, r21, r31, r41)
print("System 2 (mech 2): r12, r22, r32, r42 =", r12, r22, r32, r42)

t11 = np.deg2rad(angle_deg(sys1_links[0]))
print("t11: ", np.rad2deg(t11), " (should be about -2.252 deg)")

# def get_k(line31, line12, coupler):
#     # s stands for start
#     # e stands for end
#     x31_s, y31_s, x31_e, y31_e = line31
#     x12_s, y12_s, x12_e, y12_e = line12
#     xc_s, yc_s, xc_e, yc_e = coupler
    
#     k1_x = xc_s - x31_e
#     k1_y = yc_s - y31_e
#     k2_x =  xc_e - x12_s
#     k2_y =  yc_e - y12_s
#     l_c1 = length(coupler)
    
#     return k1_x, k1_y, k2_x, k2_y, l_c1
# # k1 defines the distance from r31 endpoint (B) to the coupler start
# # k2 defines the distance from r12 startpoint (O_2) to the coupler end
# k1_x, k1_y, k2_x, k2_y, l_c1 = get_k(sys1_links[2], sys2_links[0], coupler_links[0])

T1 = [0, 0] # offset/translation of local coordinate system 1 to global coordinate system
Phi1 = 0 # by defintion #####################################np.deg2rad(angle_deg(sys1_links[0])) # rotation of local coordinate system 1 with respect to the global coordinate system

def get_A1A2_initial(line21, line22):
    x21_s, y21_s, x21_e, y21_e = line21
    x22_s, y22_s, x22_e, y22_e = line22
    
    l_A1A2 = math.hypot(x22_e - x21_e, y22_e - y21_e)
    t_A1A2 = math.atan2(y22_e - y21_e, x22_e - x21_e)
    return l_A1A2,t_A1A2
l_A1A2, t_A1A2_initial = get_A1A2_initial(sys1_links[1], sys2_links[1])

#Phi2 = t_A1A2 # local x-axis of system two aligns with the line A1A2
t31_initial = np.deg2rad(angle_deg(sys1_links[2]))
t32_initial = np.deg2rad(angle_deg(sys2_links[2]))
t_constant = (2*np.pi - t31_initial) + t_A1A2_initial # constant value between due to rigid part connecting r31 and r32
Phi2 = t_constant - (2*np.pi - t31)

t32 = t32_initial - t31_initial + t31 - Phi2 # derived from t32 - t31 = const and converting to the local frame by -Phi2
#t32 = math.radians(angle_deg(sys2_links[2])) - Phi2#t32 = t31 + np.deg2rad( angle_deg(sys2_links[2]) - angle_deg(sys1_links[2]) ) #################################


def compute_k_offsets(sys1_links, sys2_links, coupler, Phi2):
    """
    sys1_links: [r1, r2, r3, r4] for system 1 (in global coords)
    sys2_links: [r1, r2, r3, r4] for system 2 (in global coords)
    coupler: [xc_s, yc_s, xc_e, yc_e] for connector between the two systems (in global coords)
    Phi2: orientation of system 2 local x-axis in global frame
    """

    # --- System 1: use endpoint of r31 as B1, coupler start as C1 ---
    x41_s, y41_s, x41_e, y41_e = sys1_links[3]         # r41 top link of system 1
    xc_s, yc_s, xc_e, yc_e = coupler                   # coupler start/end

    B1_world = np.array([x41_e, y41_e], dtype=float)   # endpoint of r41
    P1_world = np.array([xc_s, yc_s], dtype=float)     # coupler start

    k1_world = P1_world - B1_world                     # in global frame
    k1_x, k1_y = k1_world                              # system 1 local == global

    # --- System 2: use start of r12 as O2_2, coupler end as C2 ---
    x12_s, y12_s, x12_e, y12_e = sys2_links[0]         # r12 bottom link of system 2
    O2_2_world = np.array([x12_s, y12_s], dtype=float)
    P2_world   = np.array([xc_e, yc_e], dtype=float)   # coupler end

    k2_world = P2_world - O2_2_world                   # vector in global frame

    # Convert to system 2 local coordinates: k2_local = R(-Phi2) * k2_world
    cos_m = math.cos(-Phi2)
    sin_m = math.sin(-Phi2)
    R_minus = np.array([[cos_m, -sin_m],
                        [sin_m,  cos_m]])
    k2_local = R_minus @ k2_world
    k2_x, k2_y = k2_local

    # Coupler length
    l_c1 = length(coupler)

    return k1_x, k1_y, k2_x, k2_y, l_c1
# k1 defines the distance from r31 endpoint (B) to the coupler start in LOCAL frame
# k2 defines the distance from r12 startpoint (O_2) to the coupler end in LOCAL frame
k1_x, k1_y, k2_x, k2_y, l_c1 = compute_k_offsets(sys1_links, sys2_links, coupler_links[0], t_A1A2_initial)



# ==== Kinematic Calculations ================================================
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
t32_val = float(t32.subs({t41: t41_val, t31: t31_val}))
Phi2_val = float(Phi2.subs({t31: t31_val}))
result_sys1 = {t41: t41_val, t31: t31_val, t32: t32_val}

eqs_sys2 = []
# Local vector loop in system 2
eqs_sys2.append( r22*sp.cos(t22) + r32*sp.cos(t32) - (r12*sp.cos(t12) + r42*sp.cos(t42)) )
eqs_sys2.append( r22*sp.sin(t22) + r32*sp.sin(t32) - (r12*sp.sin(t12) + r42*sp.sin(t42)) )
####### add explanation for the conversion
# Vector loop equation incorporating the connecting link ###############################################################################################################################
#################################################
##########################################
##########################################
eqs_sys2.append( r21*sp.cos(t21) + l_A1A2*sp.cos(Phi2) - r22*sp.cos(t22 + Phi2) + ( k2_x*sp.cos(Phi2) - k2_y*sp.sin(Phi2) ) - l_c1*sp.cos(alpha1) - ( r41*(1)*sp.cos(t41) + k1_x ) - r11*sp.cos(t11)) # note that [k2_x, k2_y] is converted using relations knowen from polar coordinates 
eqs_sys2.append( r21*sp.sin(t21) + l_A1A2*sp.sin(Phi2) - r22*sp.sin(t22 + Phi2) + ( k2_x*sp.sin(Phi2) + k2_y*sp.cos(Phi2) ) - l_c1*sp.sin(alpha1) - ( r41*(1)*sp.sin(t41) + k1_y ) - r11*sp.sin(t11)) # note that [k2_x, k2_y] is converted using relations knowen from polar coordinates
#eqs_sys2.append( r21*sp.cos(t21) + l_A1A2*sp.cos(t_A1A2) - r22*sp.cos(t22 + Phi2) + ( k2_x*sp.cos(Phi2) - k2_y*sp.sin(Phi2) ) - l_c1*sp.cos(alpha1) - r41*(1+k1)*sp.cos(t41) ) # note that [k2_x, k2_y] is converted using relations knowen from polar coordinates 
#eqs_sys2.append( r21*sp.sin(t21) + l_A1A2*sp.sin(t_A1A2) - r22*sp.sin(t22 + Phi2) + ( k2_x*sp.sin(Phi2) + k2_y*sp.cos(Phi2) ) - l_c1*sp.sin(alpha1) - r41*(1+k1)*sp.sin(t41) ) # note that [k2_x, k2_y] is converted using relations knowen from polar coordinates


eqs_sys2 = [e.subs(result_sys1) for e in eqs_sys2]


# x0_sys2 = [
#     np.deg2rad(0),    # t12
#     np.deg2rad(86.1),  # t22
#     np.deg2rad(82),   # t42
#     np.deg2rad(-1),   # alpha1
# ]
x0_sys2 = [
    math.radians(angle_deg(sys2_links[0])) - Phi2_val,    # t12
    math.radians(angle_deg(sys2_links[1])) - Phi2_val,  # t22
    math.radians(angle_deg(sys2_links[3])) - Phi2_val,   # t42
    math.radians(angle_deg(coupler_links[0])),   # alpha1
]
unknowns_sys2 = [t12, t22, t42, alpha1]

########### do not solve system but plot it if solve_numerically == False
solve_numerically = True
if solve_numerically:
    sol_sys2 = sp.nsolve(eqs_sys2, unknowns_sys2, x0_sys2, tol=1e-14, maxsteps=100)
    t12_val, t22_val, t42_val, alpha1_val = map(float, sol_sys2)
else:
    t12_val, t22_val, t42_val, alpha1_val = x0_sys2



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
P1_loc = B1_loc + sp.Matrix([k1_x, k1_y]) ### here specified as an offset to B1_loc by k1_x, k1_y

# Startpoint of r22
O2_2_loc = sp.Matrix([0,0])
# Endpoint of r22
A2_loc = sp.Matrix([r22*sp.cos(t22), r22*sp.sin(t22)])
# Startpoint of r42
O4_2_loc = sp.Matrix([r12*sp.cos(t12), r12*sp.sin(t12)])
# Endpoint of r42
B2_loc = O4_2_loc + sp.Matrix([r42*sp.cos(t42), r42*sp.sin(t42)])
# Fixation point for connecting linkage to system 2
#P2_loc = O2_2_loc + sp.Matrix([k2_x, k2_y]) ### here specified as an offset to O2_loc (origin) by k2_x and k2_y
P2_loc = sp.Matrix([k2_x, k2_y]) ### here specified as an offset to O2_loc (origin) by k2_x and k2_y



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
A2_global = A1_global + l_A1A2 * np.array([np.cos(Phi2_val), np.sin(Phi2_val)], dtype=float)
# We know for every point P holds P =  O2_2_global + R2 @ P_local (translation and rotation)
# Plug in A2 and rearange to get:
############A2_global = O2_2_global + R(Phi2_val) @ to_np(A2_loc)
O2_2_global = A2_global - R(Phi2_val) @ to_np(A2_loc) 
# And use the expression for P to evaluate the remaining global coordinates
O4_2_global = O2_2_global + R(Phi2_val) @ to_np(O4_2_loc)
B2_global   = O2_2_global + R(Phi2_val) @ to_np(B2_loc) 
P2_global   = O2_2_global + R(Phi2_val) @ to_np(P2_loc)


### Sanity checks
print("alpha1: ", alpha1_val)
print("t11 t21 t31 t41: ", np.rad2deg(t11), np.rad2deg(t21), np.rad2deg(t31_val), np.rad2deg(t41_val))
print("local: t12 t22 t32 t42: ", np.rad2deg(t12_val), np.rad2deg(t22_val), np.rad2deg(t32_val), np.rad2deg(t42_val))
print("global: t12 t22 t32 t42: ", np.rad2deg(t12_val  + Phi2_val), np.rad2deg(t22_val + Phi2_val), np.rad2deg(t32_val + Phi2_val), np.rad2deg(t42_val + Phi2_val))
conn_len = np.linalg.norm(P2_global - P1_global)
print(f"[Check] |P2 - P1| = {conn_len:.6f} (target l_c = {length(coupler_links[0]):.6f})") ####################################
print(f"[Check] |B1 - A1| = {np.linalg.norm(B1_global - A1_global):.6f} (target r31 = {r31:.6f})")
print(f"[Check] |B2 - A2| = {np.linalg.norm(B2_global - A2_global):.6f} (target r32 = {r32:.6f})")

########################
# sanity check
plt.figure()
plt.title("Second four-bar in local coordinates")
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
