import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import itertools

# ================================================================
# 1) INPUT GEOMETRY
#    four_bar_linkages can be in random order
#    coupler_links must be ordered left->right and oriented left->right
# ================================================================

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

# ================================================================
# 2) GEOMETRY HELPERS + LINK ORDERING
# ================================================================

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
    """
    if len(linkage) != 4:
        raise ValueError("Each four-bar linkage must have exactly 4 links.")

    lines = [list(line) for line in linkage]
    mids  = [midpoint(line) for line in lines]

    ys = [m[1] for m in mids]
    min_y = min(ys); max_y = max(ys)
    tol_y = 0.1 * (max_y - min_y + 1e-9)

    bottom_indices = [i for i, (xm, ym) in enumerate(mids) if ym <= min_y + tol_y]
    top_indices    = [i for i, (xm, ym) in enumerate(mids) if ym >= max_y - tol_y]

    if not bottom_indices:
        bottom_indices = [ys.index(min_y)]
    if not top_indices:
        top_indices = [ys.index(max_y)]

    def horiz_score(i):
        a = angle_deg(lines[i])
        a_mod = abs(((a + 90) % 180) - 90)
        return a_mod

    r1_idx = min(bottom_indices, key=horiz_score)
    r3_idx = min(top_indices, key=horiz_score)

    remaining = [i for i in range(4) if i not in (r1_idx, r3_idx)]
    if len(remaining) != 2:
        raise RuntimeError("Side link detection failed; check geometry.")

    x_remaining = [mids[i][0] for i in remaining]
    left_idx = remaining[x_remaining.index(min(x_remaining))]
    right_idx = remaining[x_remaining.index(max(x_remaining))]
    r2_idx, r4_idx = left_idx, right_idx

    r1 = orient_left_to_right(lines[r1_idx])
    r3 = orient_left_to_right(lines[r3_idx])
    r2 = orient_bottom_to_top(lines[r2_idx])
    r4 = orient_bottom_to_top(lines[r4_idx])

    return [r1, r2, r3, r4]

def sort_mechanisms_left_to_right(ordered_linkages):
    """
    Sort the four-bar mechanisms themselves from left to right,
    based on the midpoint x of their bottom (r1) link.
    """
    centers_x = []
    for links in ordered_linkages:
        r1 = links[0]  # bottom ground link
        x1, y1, x2, y2 = r1
        centers_x.append(0.5 * (x1 + x2))

    idx_sorted = sorted(range(len(ordered_linkages)), key=lambda i: centers_x[i])
    ordered_lr = [ordered_linkages[i] for i in idx_sorted]
    return ordered_lr, idx_sorted

ordered_linkages, idx_sorted = sort_mechanisms_left_to_right([order_links_rectangular(l) for l in four_bar_linkages])
# lengths for all mechanisms
r = [[length(line) for line in links] for links in ordered_linkages]###################

print("=== Ordered four-bar linkages (r1..r4 CCW) ===\n")
for i, ordered in enumerate(ordered_linkages):
    print(f"Mechanism {i+1}:")
    for j, line in enumerate(ordered, start=1):
        name = f"r{j}{i+1}"
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
    L = length(line)
    ang = angle_deg(line)
    x1, y1, x2, y2 = line
    print(
        f"  {name}: start=({x1:.3f}, {y1:.3f}), "
        f"end=({x2:.3f}, {y2:.3f}), "
        f"length={L:.3f}, angle={ang:.3f} deg"
    )

# ================================================================
# 3) CHOOSE WHICH TWO FOUR-BARS TO COUPLE (system 1 & 2)
#    Here: system 1 = mechanism 1, system 2 = mechanism 2, connector = coupler_links[0]
# ================================================================

sys1_links = ordered_linkages[0]  # [r11, r21, r31, r41]
sys2_links = ordered_linkages[1]  # [r12, r22, r32, r42]
connector  = coupler_links[0]

r11, r21, r31, r41 = [length(line) for line in sys1_links]
r12, r22, r32, r42 = [length(line) for line in sys2_links]
t11 = math.radians(angle_deg(sys1_links[0]))  # ground angle of system 1

print("\nLengths from geometry:")
print("System 1 (mech 1): r11, r21, r31, r41 =", r11, r21, r31, r41)
print("System 2 (mech 2): r12, r22, r32, r42 =", r12, r22, r32, r42)
print("t11 (deg):", math.degrees(t11))

# ================================================================
# 3a) GEOMETRY FOR MECHANISMS 3 AND 4 (LIKE 1 & 2)
# ================================================================
sys3_links = ordered_linkages[2]  # [r13, r23, r33, r43]
sys4_links = ordered_linkages[3]  # [r14, r24, r34, r44]
connector_34 = coupler_links[2]

r13, r23, r33, r43 = [length(line) for line in sys3_links]
r14, r24, r34, r44 = [length(line) for line in sys4_links]

t13 = math.radians(angle_deg(sys3_links[0]))  # ground angle of system 3

# ================================================================
# 3b) GEOMETRY FOR THE 2-3 BOTTOM BRACKET + COUPLER 2
#       and solver
# ================================================================
connector_23 = coupler_links[1]
# --- geometry for coupler 2 with offsets to O4_2 and A3 ---

# link lengths for mechanism 3
r13, r23, r33, r43 = r[2]

# CAD points for mechanism 2 & 3 we care about

# O4_2_CAD = bottom-right joint of mech 2 (end of r1_2)
x12_s, y12_s, x12_e, y12_e = sys2_links[0]  # r1_2
O4_2_CAD = np.array([x12_e, y12_e], float)

# O2_3_CAD: bottom-left ground joint of mech 3 (start of r1_3)
x13_s, y13_s, x13_e, y13_e = sys3_links[0]  # r1_3
O2_3_CAD = np.array([x13_s, y13_s], float)

# A3_CAD: top-left moving joint of mech 3 (end of r2_3)
x23_s, y23_s, x23_e, y23_e = sys3_links[1]  # r2_3: O2_3 -> A3
A3_CAD   = np.array([x23_e, y23_e], float)

# CAD coordinates of coupler 2 endpoints
xc2_s, yc2_s, xc2_e, yc2_e = connector_23
C2L_CAD = np.array([xc2_s, yc2_s], float)  # left end (near O4_2)
C2R_CAD = np.array([xc2_e, yc2_e], float)  # right end (near A3)

# bottom bracket vector (O4_2 -> O2_3)
br23_vec = O2_3_CAD - O4_2_CAD

# ===== Offsets for coupler 2 =====

# 1) offset from O4_2 to left coupler joint (in world frame)
k2L_world = C2L_CAD - O4_2_CAD
k2L_x, k2L_y = k2L_world

# 2) offset from A3 to right coupler joint, expressed in the *body-fixed* frame of link r23
#    (so that it rotates with r23 about A3)
deltaR_world = C2R_CAD - A3_CAD
t23_CAD_global = math.radians(angle_deg(sys3_links[1]))  # orientation of r23 in CAD

def R2(phi):
    return np.array([[math.cos(phi), -math.sin(phi)],
                     [math.sin(phi),  math.cos(phi)]])

# local offset in r23's frame
deltaR_local = R2(-t23_CAD_global) @ deltaR_world
deltaR_loc_x, deltaR_loc_y = deltaR_local

# coupler length
l_c2 = length(connector_23)

print("Coupler 2 offsets:")
print("  k2L_world =", k2L_world)
print("  deltaR_local (right-side offset in r23 frame) =", deltaR_local)
print("  l_c2 =", l_c2)

# unknowns for mechanism 3 + coupler 2
t23_sym, t33_sym_23, t43_sym_23, alpha2_sym = sp.symbols(
    't23 t33_23 t43_23 alpha2', real=True
)

# we keep the ground link r13 at its CAD orientation
t13_const = math.radians(angle_deg(sys3_links[0]))  # orientation of r13 in world frame

def solve_pair23(O4_2_global, prev_guess=None):
    """
    Solve mechanism 3 + coupler 2 for a given O4_2_global.

    - Mechanism 3 is a four-bar with ground link r13 at fixed angle t13_const,
      origin at O2_3_global = O4_2_global + br23_vec.
    - Coupler 2 is rigidly attached:
        left: offset k2L_world from O4_2
        right: offset deltaR_local from A3, rotating with link r23
    """

    # global position of O2_3 from bottom bracket
    O2_3_global = O4_2_global + br23_vec

    # Four-bar geometry in world frame
    # O2_3: origin of mech 3 ground
    # O4_3 = O2_3 + r13 * [cos t13_const, sin t13_const]
    O4_3x = O2_3_global[0] + r13 * math.cos(t13_const)
    O4_3y = O2_3_global[1] + r13 * math.sin(t13_const)

    # A3 position (depends on t23)
    A3x = O2_3_global[0] + r23 * sp.cos(t23_sym)
    A3y = O2_3_global[1] + r23 * sp.sin(t23_sym)

    # B3 position (depends on t43)
    B3x = O4_3x + r43 * sp.cos(t43_sym_23)
    B3y = O4_3y + r43 * sp.sin(t43_sym_23)

    # Four-bar closure: B3 - A3 = r33 * [cos t33, sin t33]
    eq_4bar_x = B3x - A3x - r33 * sp.cos(t33_sym_23)
    eq_4bar_y = B3y - A3y - r33 * sp.sin(t33_sym_23)

    # Coupler 2 endpoints:

    # left end (fixed offset from O4_2)
    C2Lx = O4_2_global[0] + k2L_x
    C2Ly = O4_2_global[1] + k2L_y

    # right end: offset from A3, rotating with link r23
    # current global angle of r23 = t23_sym
    # deltaR_local is expressed in r23's body frame at CAD
    # so we rotate it by t23_sym to get world offset
    offRx = deltaR_loc_x * sp.cos(t23_sym) - deltaR_loc_y * sp.sin(t23_sym)
    offRy = deltaR_loc_x * sp.sin(t23_sym) + deltaR_loc_y * sp.cos(t23_sym)

    C2Rx = A3x + offRx
    C2Ry = A3y + offRy

    # Coupler length/orientation constraints:
    # C2R - C2L = l_c2 * [cos alpha2, sin alpha2]
    eq_coup_x = C2Rx - C2Lx - l_c2 * sp.cos(alpha2_sym)
    eq_coup_y = C2Ry - C2Ly - l_c2 * sp.sin(alpha2_sym)

    eqs3 = [eq_4bar_x, eq_4bar_y, eq_coup_x, eq_coup_y]

    if prev_guess is None:
        # use CAD-ish angles as start
        t23_0 = math.radians(angle_deg(sys3_links[1]))
        t33_0 = math.radians(angle_deg(sys3_links[2]))
        t43_0 = math.radians(angle_deg(sys3_links[3]))
        alpha2_0 = math.radians(angle_deg(connector_23))
        prev_guess = (t23_0, t33_0, t43_0, alpha2_0)

    sol3 = sp.nsolve(
        eqs3, (t23_sym, t33_sym_23, t43_sym_23, alpha2_sym),
        prev_guess, tol=1e-12, maxsteps=100
    )

    t23_val, t33_val, t43_val, alpha2_val = map(float, sol3)

    angles23 = dict(
        t23=t23_val,
        t33=t33_val,
        t43=t43_val,
        alpha2=alpha2_val,
        # we can reconstruct Phi3, etc., later if needed
    )
    new_guess = (t23_val, t33_val, t43_val, alpha2_val)
    return angles23, new_guess


# ================================================================
# 4) SYMBOLIC VARIABLES & GEOMETRIC RELATIONS
# ================================================================

# rotation helper
def R(phi):
    return np.array([[math.cos(phi), -math.sin(phi)],
                     [math.sin(phi),  math.cos(phi)]])

# Sympy symbols
t31, t41, t12, t22, t42, alpha1 = sp.symbols('t31 t41 t12 t22 t42 alpha1', real=True)

# Input angle of link 2 of system 1 will be varied later (t21)
# (we treat t21 as numeric parameter inside the solver)

def get_A1A2_initial(line21, line22):
    x21_s, y21_s, x21_e, y21_e = line21
    x22_s, y22_s, x22_e, y22_e = line22
    l_A1A2 = math.hypot(x22_e - x21_e, y22_e - y21_e)
    t_A1A2 = math.atan2(y22_e - y21_e, x22_e - x21_e)
    return l_A1A2, t_A1A2

l_A1A2, t_A1A2_initial = get_A1A2_initial(sys1_links[1], sys2_links[1])


t31_initial = math.radians(angle_deg(sys1_links[2]))  # global r31 angle
t32_initial = math.radians(angle_deg(sys2_links[2]))  # global r32 angle

# rigid relation between r31, A1A2, and Phi2:
# Phi2 = const - (2*pi - t31)
t_constant = (2 * math.pi - t31_initial) + t_A1A2_initial
Phi2_expr = t_constant - (2 * math.pi - t31)

# relation between t31 and local t32 (in system 2 frame)
# t32_global - t31_global = const  ⇒ t32_expr = (t32_initial - t31_initial) + t31,
# then convert to local frame by subtracting Phi2
t32_expr = (t32_initial - t31_initial) + t31 - Phi2_expr

def compute_k_offsets(sys1_links, sys2_links, coupler, Phi2_val):
    """
    sys1_links: [r1, r2, r3, r4] for system 1 (global coords from CAD)
    sys2_links: [r1, r2, r3, r4] for system 2 (global coords from CAD)
    coupler:   connector between them in global coords
    Phi2_val:  orientation of system 2 local x-axis in global frame
               (here: use initial Phi2 = t_A1A2_initial so k2 is defined
               in the initial local frame)
    """
    # System 1: use endpoint of r41 as B1, coupler start as P1
    x41_s, y41_s, x41_e, y41_e = sys1_links[3]  # r41
    xc_s, yc_s, xc_e, yc_e = coupler
    B1_world = np.array([x41_e, y41_e], float)
    P1_world = np.array([xc_s, yc_s], float)
    k1_world = P1_world - B1_world
    k1_x, k1_y = k1_world  # local = global for system 1

    # System 2: use start of r12 as O2_2, coupler end at P2
    x12_s, y12_s, x12_e, y12_e = sys2_links[0]
    O2_2_world = np.array([x12_s, y12_s], float)
    P2_world   = np.array([xc_e, yc_e], float)
    k2_world   = P2_world - O2_2_world

    # convert to system 2 local via rotation -Phi2
    cos_m = math.cos(-Phi2_val)
    sin_m = math.sin(-Phi2_val)
    R_minus = np.array([[cos_m, -sin_m],
                        [sin_m,  cos_m]])
    k2_local = R_minus @ k2_world
    k2_x, k2_y = k2_local

    l_c1 = length(coupler)

    return k1_x, k1_y, k2_x, k2_y, l_c1

# k1: in local frame of system 1 (== global)
# k2: in local frame of system 2 at initial configuration
k1_x, k1_y, k2_x, k2_y, l_c1 = compute_k_offsets(
    sys1_links, sys2_links, connector, t_A1A2_initial
)

print("\nConnector data from CAD in local form:")
print("k1_x, k1_y =", k1_x, k1_y)
print("k2_x, k2_y =", k2_x, k2_y)
print("l_c1      =", l_c1)

# ================================================================
# 4a) GEOMETRY FOR MECHANISMS 3 AND 4 (LIKE 1 & 2)
#       and define solver here which will be called after system 3 was solved
# ================================================================
# System 3 to 4
l_A3A4, t_A3A4_initial = get_A1A2_initial(sys3_links[1], sys4_links[1])

t33_initial = math.radians(angle_deg(sys3_links[2]))
t34_initial = math.radians(angle_deg(sys4_links[2]))

# same kind of relation as for Phi2 and t32_expr
t_constant_34 = (2 * math.pi - t33_initial) + t_A3A4_initial

# symbolic angles for pair 3-4
t33_sym, t43_sym, t14_sym, t24_sym, t44_sym, alpha3_sym = sp.symbols(
    't33 t43 t14 t24 t44 alpha3', real=True
)

# Phi4 at the initial configuration from the CAD
Phi4_initial = t_A3A4_initial
k3L_x, k3L_y, k3R_x, k3R_y, l_c3 = compute_k_offsets(
    sys3_links, sys4_links, connector_34, Phi4_initial
)

print("3-4 connector data:")
print("k3L_x, k3L_y =", k3L_x, k3L_y)
print("k3R_x, k3R_y =", k3R_x, k3R_y)
print("l_c3 =", l_c3)

def solve_pair34(t33_val, t23_val, t43_val, prev_guess=None):
    """
    Solve four-bar 4 and coupler 3 for a given configuration of mechanism 3.
    Input:
        t33_val: global angle of r33 (top link of mech 3)
        t23_val: local angle of r23 in system 3 frame
        t43_val: local angle of r43 in system 3 frame
    Returns:
        dict of angles for mech 4 and coupler 3 + updated initial guess.
    """
    # --- bracket relation: Phi4 from t33, like Phi2 from t31 ---
    Phi4_val = t_constant_34 - (2 * math.pi - t33_val)

    # t34_global = t34_initial + (t33 - t33_initial)
    t34_global = t34_initial + (t33_val - t33_initial)
    # local angle of r34 in frame 4
    t34_local = t34_global - Phi4_val

    # --- system 4 equations (four-bar + connector) ---
    eqs4 = [
        # local four-bar loop in system 4
        r24 * sp.cos(t24_sym) + r34 * math.cos(t34_local) \
            - (r14 * sp.cos(t14_sym) + r44 * sp.cos(t44_sym)),
        r24 * sp.sin(t24_sym) + r34 * math.sin(t34_local) \
            - (r14 * sp.sin(t14_sym) + r44 * sp.sin(t44_sym)),

        # connector loop, x
        r23 * math.cos(t23_val) + l_A3A4 * math.cos(Phi4_val) \
            - r24 * sp.cos(t24_sym + Phi4_val) \
            + (k3R_x * math.cos(Phi4_val) - k3R_y * math.sin(Phi4_val)) \
            - l_c3 * sp.cos(alpha3_sym) \
            - (r13 * math.cos(t13) + r43 * math.cos(t43_val) + k3L_x),

        # connector loop, y
        r23 * math.sin(t23_val) + l_A3A4 * math.sin(Phi4_val) \
            - r24 * sp.sin(t24_sym + Phi4_val) \
            + (k3R_x * math.sin(Phi4_val) + k3R_y * math.cos(Phi4_val)) \
            - l_c3 * sp.sin(alpha3_sym) \
            - (r13 * math.sin(t13) + r43 * math.sin(t43_val) + k3L_y),
    ]

    if prev_guess is None:
        prev_guess = (
            math.radians(angle_deg(sys4_links[0])) - Phi4_initial,  # t14
            math.radians(angle_deg(sys4_links[1])) - Phi4_initial,  # t24
            math.radians(angle_deg(sys4_links[3])) - Phi4_initial,  # t44
            math.radians(angle_deg(connector_34)),                  # alpha3
        )

    sol4 = sp.nsolve(eqs4, (t14_sym, t24_sym, t44_sym, alpha3_sym),
                     prev_guess, tol=1e-12, maxsteps=100)

    t14_val, t24_val, t44_val, alpha3_val = map(float, sol4)

    angles34 = dict(
        Phi4=Phi4_val,
        t34=t34_local,
        t14=t14_val,
        t24=t24_val,
        t44=t44_val,
        alpha3=alpha3_val,
    )
    new_guess = (t14_val, t24_val, t44_val, alpha3_val)
    return angles34, new_guess

# ================================================================
# 5) SOLVER FUNCTIONS (SYSTEM 1 & 2)
# ================================================================

def solve_mechanism(t21_val, prev_guess_sys2=None):
    """
    Solve both four-bars for a given input angle t21 (rad).
    Returns:
      angles dict and updated guess for system 2.
    """
    # ----- System 1: four-bar with t21 as input -----
    eqs1 = [
        r21 * sp.cos(t21_val) + r31 * sp.cos(t31) - (r11 * sp.cos(t11) + r41 * sp.cos(t41)),
        r21 * sp.sin(t21_val) + r31 * sp.sin(t31) - (r11 * sp.sin(t11) + r41 * sp.sin(t41)),
    ]

    # Tweak this initial guess if needed
    x0_sys1 = (math.radians(20), math.radians(10))
    sol1 = sp.nsolve(eqs1, (t41, t31), x0_sys1)
    t41_val = float(sol1[0])
    t31_val = float(sol1[1])

    # Derived quantities
    Phi2_val = float(Phi2_expr.subs({t31: t31_val}))
    t32_val  = float(t32_expr.subs({t31: t31_val}))

    # ----- System 2: four-bar + connector constraint -----
    eqs2 = [
        # local four-bar loop in system 2
        r22 * sp.cos(t22) + r32 * math.cos(t32_val) - (r12 * sp.cos(t12) + r42 * sp.cos(t42)),
        r22 * sp.sin(t22) + r32 * math.sin(t32_val) - (r12 * sp.sin(t12) + r42 * sp.sin(t42)),
        # r22 * sp.cos(t22) + r32 * math.cos(t32_val - Phi2_val) - (r12 * sp.cos(t12) + r42 * sp.cos(t42)),
        # r22 * sp.sin(t22) + r32 * math.sin(t32_val - Phi2_val) - (r12 * sp.sin(t12) + r42 * sp.sin(t42)),
        # connector loop, x-component
        r21 * math.cos(t21_val) + l_A1A2 * math.cos(Phi2_val) - r22 * sp.cos(t22 + Phi2_val) \
        + (k2_x * math.cos(Phi2_val) - k2_y * math.sin(Phi2_val)) \
        - l_c1 * sp.cos(alpha1) - (r11 * math.cos(t11) + r41 * math.cos(t41_val) + k1_x),
        # connector loop, y-component
        r21 * math.sin(t21_val) + l_A1A2 * math.sin(Phi2_val) - r22 * sp.sin(t22 + Phi2_val) \
        + (k2_x * math.sin(Phi2_val) + k2_y * math.cos(Phi2_val)) \
        - l_c1 * sp.sin(alpha1) - (r11 * math.sin(t11) + r41 * math.sin(t41_val) + k1_y),
    ]

    # Initial guess for system 2
    if prev_guess_sys2 is None:
        prev_guess_sys2 = (
            math.radians(angle_deg(sys2_links[0])) - Phi2_val,  # t12
            math.radians(angle_deg(sys2_links[1])) - Phi2_val,  # t22
            math.radians(angle_deg(sys2_links[3])) - Phi2_val,  # t42
            math.radians(angle_deg(connector)),                 # alpha1
        )

    sol2 = sp.nsolve(eqs2, (t12, t22, t42, alpha1),
                     prev_guess_sys2, tol=1e-12, maxsteps=100)
    t12_val, t22_val, t42_val, alpha1_val = map(float, sol2)

    angles = dict(
        t21=t21_val,
        t31=t31_val, t41=t41_val,
        t12=t12_val, t22=t22_val, t32=t32_val,
        t42=t42_val,
        Phi2=Phi2_val,
        alpha1=alpha1_val,
    )

    # use this as initial guess for next configuration
    new_guess_sys2 = (t12_val, t22_val, t42_val, alpha1_val)

    return angles, new_guess_sys2

# ================================================================
# 6) POSITION EVALUATION FOR A GIVEN SET OF ANGLES
# ================================================================

def compute_positions(angles):
    """
    Given solved angles, compute global coordinates of all joints and connector.
    Returns dict with points:
        O2_1, A1, O4_1, B1, P1,
        O2_2, A2, O4_2, B2, P2
    """
    t21_val = angles["t21"]
    t31_val = angles["t31"]
    t41_val = angles["t41"]
    t12_val = angles["t12"]
    t22_val = angles["t22"]
    t32_val = angles["t32"]
    t42_val = angles["t42"]
    Phi2_val = angles["Phi2"]

    # --- System 1 local (same as global) ---
    O2_1 = np.array([0.0, 0.0])
    A1   = O2_1 + np.array([r21 * math.cos(t21_val),
                            r21 * math.sin(t21_val)])
    O4_1 = np.array([r11 * math.cos(t11),
                     r11 * math.sin(t11)])
    B1   = O4_1 + np.array([r41 * math.cos(t41_val),
                            r41 * math.sin(t41_val)])
    P1   = B1 + np.array([k1_x, k1_y])  # offset in global frame

    # --- System 2 local coordinates ---
    O2_2_loc = np.array([0.0, 0.0])
    A2_loc   = O2_2_loc + np.array([r22 * math.cos(t22_val),
                                    r22 * math.sin(t22_val)])
    O4_2_loc = np.array([r12 * math.cos(t12_val),
                         r12 * math.sin(t12_val)])
    B2_loc   = O4_2_loc + np.array([r42 * math.cos(t42_val),
                                    r42 * math.sin(t42_val)])
    P2_loc   = np.array([k2_x, k2_y])

    # --- Transform system 2 local -> global ---
    A2_global = A1 + l_A1A2 * np.array([math.cos(Phi2_val),
                                        math.sin(Phi2_val)])

    # A2_global = O2_2_global + R(Phi2) @ A2_loc  ⇒  O2_2_global = ...
    O2_2 = A2_global - R(Phi2_val) @ A2_loc
    O4_2 = O2_2 + R(Phi2_val) @ O4_2_loc
    B2   = O2_2 + R(Phi2_val) @ B2_loc
    P2   = O2_2 + R(Phi2_val) @ P2_loc

    return dict(
        O2_1=O2_1, A1=A1, O4_1=O4_1, B1=B1, P1=P1,
        O2_2=O2_2, A2=A2_global, O4_2=O4_2, B2=B2, P2=P2
    )

def compute_positions_mech3(angles23, O4_2_global, br23_vec):
    """
    Compute global coordinates for mechanism 3 and coupler 2 attachment,
    consistent with solve_pair23.
    """
    t23 = angles23["t23"]
    t33 = angles23["t33"]
    t43 = angles23["t43"]

    # mechanism 3 lengths
    r13, r23, r33, r43 = r[2]

    # ground angle of r13 is fixed:
    # (we already defined this above as t13_const)
    O2_3 = O4_2_global + br23_vec
    O4_3 = O2_3 + np.array([r13 * math.cos(t13_const),
                            r13 * math.sin(t13_const)])
    A3   = O2_3 + np.array([r23 * math.cos(t23),
                            r23 * math.sin(t23)])
    B3   = O4_3 + np.array([r43 * math.cos(t43),
                            r43 * math.sin(t43)])

    # coupler 2 endpoints (same model as in solve_pair23)
    C2L = O4_2_global + k2L_world  # left end
    offR = np.array([
        deltaR_loc_x * math.cos(t23) - deltaR_loc_y * math.sin(t23),
        deltaR_loc_x * math.sin(t23) + deltaR_loc_y * math.cos(t23),
    ])
    C2R = A3 + offR  # right end
    
    P3 = B3 + np.array([k3L_x, k3L_y])

    return dict(
        O2_3=O2_3,
        A3=A3,
        O4_3=O4_3,
        B3=B3,
        C2L=C2L,
        C2R=C2R,
        P3=P3,
    )


def compute_positions_mech4(angles34, pts3):
    """
    Compute global coordinates for mechanism 4 and coupler 3.
    """
    Phi4 = angles34["Phi4"]
    t24 = angles34["t24"]
    t34 = angles34["t34"]  # local to mech 4 frame
    t44 = angles34["t44"]

    # link lengths for mechanism 4
    r14, r24, r34, r44 = r[3]

    # anchoring: A3 is known from mechanism 3
    A3 = pts3["A3"]
    A4 = A3 + l_A3A4 * np.array([math.cos(Phi4), math.sin(Phi4)])

    # O2_4 from A4 and local A4_loc
    A4_loc = np.array([r24 * math.cos(t24), r24 * math.sin(t24)])
    O2_4 = A4 - R(Phi4) @ A4_loc

    t14 = angles34["t14"]######################################
    O4_4_loc = np.array([r14 * math.cos(t14), r14 * math.sin(t14)])
    B4_loc = O4_4_loc + np.array([r44 * math.cos(t44), r44 * math.sin(t44)])

    O4_4 = O2_4 + R(Phi4) @ O4_4_loc
    B4 = O2_4 + R(Phi4) @ B4_loc

    # coupler 3 attachment (P4)
    P4_loc = np.array([k3R_x, k3R_y])
    P4 = O2_4 + R(Phi4) @ P4_loc

    return dict(
        O2_4=O2_4,
        A4=A4,
        O4_4=O4_4,
        B4=B4,
        P4=P4
    )
    
# ================================================================
# 7) SOLVE AND PLOT MULTIPLE CONFIGURATIONS
# ================================================================

# --- choose input angles you want to see (in degrees) ---
t21_degs = [109.473, 100, 90, 80 , 70, 60, 50]#[70.0, 80.0 , 90.0, 100.0, 109.473, 120.0, 130.0]   # you can change this list
t21_list = [math.radians(d) for d in t21_degs]

all_results = []
guess_sys2 = None
guess_23 = None
guess_34 = None

for t21_val in t21_list:
    # 1+2 as before
    angles12, guess_sys2 = solve_mechanism(t21_val, guess_sys2)
    pts12 = compute_positions(angles12)

    # --- position of O4_2_global from pts12 ---
    O4_2_global = pts12["O4_2"]  # (you may need to expose it in compute_positions)

    # 2–3
    angles23, guess_23 = solve_pair23(O4_2_global, guess_23)

    # 3–4
    angles34, guess_34 = solve_pair34(
        t33_val=angles23["t33"],
        t23_val=angles23["t23"],
        t43_val=angles23["t43"],
        prev_guess=guess_34
    )

    # now compute positions of mechanisms 3 and 4 in global coordinates,
    # analogous to compute_positions(), using angles23["Phi3"], angles34["Phi4"], etc.
    pts3 = compute_positions_mech3(angles23, O4_2_global, br23_vec)
    pts4 = compute_positions_mech4(angles34, pts3)  # anchored on mech3

    all_results.append((angles12, pts12, angles23, pts3, angles34, pts4))
    ###all_results.append([angles12, angles23, angles34], [ pts12, pts3, pts4 ])
# ================================================================
# 8) PLOTS
# ================================================================

# --- Plot a single reference configuration (e.g. the middle one) ---
#ref_angles, ref_pts = all_results[len(all_results) // 2]
(
    ref_angles12,
    ref_pts12,
    ref_angles23,
    ref_pts3,
    ref_angles34,
    ref_pts4
) = all_results[len(all_results) // 2]
plt.figure()
plt.title("Reference configuration")

# ----- System 1 -----
plt.plot([ref_pts12["O2_1"][0], ref_pts12["O4_1"][0]],
         [ref_pts12["O2_1"][1], ref_pts12["O4_1"][1]], "-k", lw=2, label="r11")
plt.plot([ref_pts12["O2_1"][0], ref_pts12["A1"][0]],
         [ref_pts12["O2_1"][1], ref_pts12["A1"][1]], "-b", label="r21")
plt.plot([ref_pts12["O4_1"][0], ref_pts12["B1"][0]],
         [ref_pts12["O4_1"][1], ref_pts12["B1"][1]], "-g", label="r41")
plt.plot([ref_pts12["A1"][0], ref_pts12["B1"][0]],
         [ref_pts12["A1"][1], ref_pts12["B1"][1]], "--b", label="r31")

# ----- System 2 -----
plt.plot([ref_pts12["O2_2"][0], ref_pts12["O4_2"][0]],
         [ref_pts12["O2_2"][1], ref_pts12["O4_2"][1]], "-r", lw=2, label="r12")
plt.plot([ref_pts12["O2_2"][0], ref_pts12["A2"][0]],
         [ref_pts12["O2_2"][1], ref_pts12["A2"][1]], "-m", label="r22")
plt.plot([ref_pts12["O4_2"][0], ref_pts12["B2"][0]],
         [ref_pts12["O4_2"][1], ref_pts12["B2"][1]], "-c", label="r42")
plt.plot([ref_pts12["A2"][0], ref_pts12["B2"][0]],
         [ref_pts12["A2"][1], ref_pts12["B2"][1]], "--m", label="r32")

# Connector 1–2
plt.plot([ref_pts12["P1"][0], ref_pts12["P2"][0]],
         [ref_pts12["P1"][1], ref_pts12["P2"][1]], "-.", color="orange", label="c1")
plt.scatter([ref_pts12["P1"][0], ref_pts12["P2"][0]],
            [ref_pts12["P1"][1], ref_pts12["P2"][1]], s=20, color="orange")

# ----- System 3 -----
plt.plot([ref_pts3["O2_3"][0], ref_pts3["O4_3"][0]],
         [ref_pts3["O2_3"][1], ref_pts3["O4_3"][1]], "-y", lw=2, label="r13")
plt.plot([ref_pts3["O2_3"][0], ref_pts3["A3"][0]],
         [ref_pts3["O2_3"][1], ref_pts3["A3"][1]], "-g", label="r23")
plt.plot([ref_pts3["O4_3"][0], ref_pts3["B3"][0]],
         [ref_pts3["O4_3"][1], ref_pts3["B3"][1]], "-b", label="r43")
plt.plot([ref_pts3["A3"][0], ref_pts3["B3"][0]],
         [ref_pts3["A3"][1], ref_pts3["B3"][1]], "--g", label="r33")

# bottom bracket between O4_2 and O2_3 (2–3)
#plt.plot([ref_pts12["O4_2"][0], ref_pts3["O2_3"][0]], [ref_pts12["O4_2"][1], ref_pts3["O2_3"][1]], "-.", color="gray", label="bracket 2–3")

# coupler 2 (if you stored its endpoints in pts3, e.g. C2L, C2R)
if "C2L" in ref_pts3 and "C2R" in ref_pts3:
    plt.plot([ref_pts3["C2L"][0], ref_pts3["C2R"][0]],
             [ref_pts3["C2L"][1], ref_pts3["C2R"][1]], "--", color="brown", label="c2")

# ----- System 4 -----
plt.plot([ref_pts4["O2_4"][0], ref_pts4["O4_4"][0]],
         [ref_pts4["O2_4"][1], ref_pts4["O4_4"][1]], "-k", lw=2, label="r14")
plt.plot([ref_pts4["O2_4"][0], ref_pts4["A4"][0]],
         [ref_pts4["O2_4"][1], ref_pts4["A4"][1]], "-r", label="r24")
plt.plot([ref_pts4["O4_4"][0], ref_pts4["B4"][0]],
         [ref_pts4["O4_4"][1], ref_pts4["B4"][1]], "-g", label="r44")
plt.plot([ref_pts4["A4"][0], ref_pts4["B4"][0]],
         [ref_pts4["A4"][1], ref_pts4["B4"][1]], "--r", label="r34")

# coupler 3 (P3–P4) if available
if "P3" in ref_pts3 and "P4" in ref_pts4:
    plt.plot([ref_pts3["P3"][0], ref_pts4["P4"][0]],
             [ref_pts3["P3"][1], ref_pts4["P4"][1]], "-.", color="purple", label="c3")

plt.axis("equal")
plt.grid(True, ls=":")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()

# --- Plot all configurations in one figure ---
plt.figure()
plt.title("Multiple configurations of all four bars")

for idx, (angles12, pts12, angles23, pts3, angles34, pts4) in enumerate(all_results):
    alpha = 0.25 + 0.75 * idx / max(1, len(all_results) - 1)

    # -------- System 1 --------
    plt.plot([pts12["O2_1"][0], pts12["O4_1"][0]],
             [pts12["O2_1"][1], pts12["O4_1"][1]], "-k", lw=1, alpha=alpha)
    plt.plot([pts12["O2_1"][0], pts12["A1"][0]],
             [pts12["O2_1"][1], pts12["A1"][1]], "-b", alpha=alpha)
    plt.plot([pts12["O4_1"][0], pts12["B1"][0]],
             [pts12["O4_1"][1], pts12["B1"][1]], "-g", alpha=alpha)
    plt.plot([pts12["A1"][0], pts12["B1"][0]],
             [pts12["A1"][1], pts12["B1"][1]], "--b", alpha=alpha)

    # -------- System 2 --------
    plt.plot([pts12["O2_2"][0], pts12["O4_2"][0]],
             [pts12["O2_2"][1], pts12["O4_2"][1]], "-r", lw=1, alpha=alpha)
    plt.plot([pts12["O2_2"][0], pts12["A2"][0]],
             [pts12["O2_2"][1], pts12["A2"][1]], "-m", alpha=alpha)
    plt.plot([pts12["O4_2"][0], pts12["B2"][0]],
             [pts12["O4_2"][1], pts12["B2"][1]], "-c", alpha=alpha)
    plt.plot([pts12["A2"][0], pts12["B2"][0]],
             [pts12["A2"][1], pts12["B2"][1]], "--m", alpha=alpha)

    # Connector 1–2
    plt.plot([pts12["P1"][0], pts12["P2"][0]],
             [pts12["P1"][1], pts12["P2"][1]], "-.", color="orange", alpha=alpha)

    # -------- System 3 --------
    plt.plot([pts3["O2_3"][0], pts3["O4_3"][0]],
             [pts3["O2_3"][1], pts3["O4_3"][1]], "-", alpha=alpha)
    plt.plot([pts3["O2_3"][0], pts3["A3"][0]],
             [pts3["O2_3"][1], pts3["A3"][1]], "-", alpha=alpha)
    plt.plot([pts3["O4_3"][0], pts3["B3"][0]],
             [pts3["O4_3"][1], pts3["B3"][1]], "-", alpha=alpha)
    plt.plot([pts3["A3"][0], pts3["B3"][0]],
             [pts3["A3"][1], pts3["B3"][1]], "--", alpha=alpha)

    # bottom bracket 2–3
    #plt.plot([pts12["O4_2"][0], pts3["O2_3"][0]], [pts12["O4_2"][1], pts3["O2_3"][1]], ":", alpha=alpha)

    # coupler 2, if C2L/C2R exist
    if "C2L" in pts3 and "C2R" in pts3:
        plt.plot([pts3["C2L"][0], pts3["C2R"][0]],
                 [pts3["C2L"][1], pts3["C2R"][1]], "--", alpha=alpha)

    # -------- System 4 --------
    plt.plot([pts4["O2_4"][0], pts4["O4_4"][0]],
             [pts4["O2_4"][1], pts4["O4_4"][1]], "-", alpha=alpha)
    plt.plot([pts4["O2_4"][0], pts4["A4"][0]],
             [pts4["O2_4"][1], pts4["A4"][1]], "-", alpha=alpha)
    plt.plot([pts4["O4_4"][0], pts4["B4"][0]],
             [pts4["O4_4"][1], pts4["B4"][1]], "-", alpha=alpha)
    plt.plot([pts4["A4"][0], pts4["B4"][0]],
             [pts4["A4"][1], pts4["B4"][1]], "--", alpha=alpha)

    # coupler 3 P3–P4 if present
    if "P3" in pts3 and "P4" in pts4:
        plt.plot([pts3["P3"][0], pts4["P4"][0]],
                 [pts3["P3"][1], pts4["P4"][1]], "-.", alpha=alpha)

plt.axis("equal")
plt.grid(True, ls=":")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()

print(ordered_linkages)
plt.show()

# =====================================================================
# 9) SIMPLE ANIMATION 110° -> 60°
# ================================================================

# --- 9.1 Compute global axis limits from all configurations ---
xs = []
ys = []

for (angles12, pts12, angles23, pts3, angles34, pts4) in all_results:
    # System 1
    for key in ["O2_1", "A1", "O4_1", "B1", "P1"]:
        p = pts12[key]
        xs.append(p[0]); ys.append(p[1])

    # System 2
    for key in ["O2_2", "A2", "O4_2", "B2", "P2"]:
        p = pts12[key]
        xs.append(p[0]); ys.append(p[1])

    # System 3
    for key in ["O2_3", "A3", "O4_3", "B3"]:
        p = pts3[key]
        xs.append(p[0]); ys.append(p[1])
    if "C2L" in pts3 and "C2R" in pts3:
        xs.append(pts3["C2L"][0]); ys.append(pts3["C2L"][1])
        xs.append(pts3["C2R"][0]); ys.append(pts3["C2R"][1])

    # System 4
    for key in ["O2_4", "A4", "O4_4", "B4"]:
        p = pts4[key]
        xs.append(p[0]); ys.append(p[1])
    if "P4" in pts4:
        xs.append(pts4["P4"][0]); ys.append(pts4["P4"][1])

xmin, xmax = min(xs), max(xs)
ymin, ymax = min(ys), max(ys)

# add small margin
dx = (xmax - xmin) * 0.1
dy = (ymax - ymin) * 0.1
xmin -= dx; xmax += dx
ymin -= dy; ymax += dy

# --- 9.2 Create figure and line objects once ---
fig_anim, ax_anim = plt.subplots()
ax_anim.set_title("Animated mechanism (110° → 60° loop)")
ax_anim.set_aspect("equal", adjustable="box")
ax_anim.set_xlim(xmin, xmax)
ax_anim.set_ylim(ymin, ymax)
ax_anim.grid(True, ls=":")
ax_anim.set_xlabel("x")
ax_anim.set_ylabel("y")

# System 1 lines
(line_r11,) = ax_anim.plot([], [], "-k", lw=2, label="r11")
(line_r21,) = ax_anim.plot([], [], "-b", label="r21")
(line_r41,) = ax_anim.plot([], [], "-g", label="r41")
(line_r31,) = ax_anim.plot([], [], "--b", label="r31")

# System 2 lines
(line_r12,) = ax_anim.plot([], [], "-r", lw=2, label="r12")
(line_r22,) = ax_anim.plot([], [], "-m", label="r22")
(line_r42,) = ax_anim.plot([], [], "-c", label="r42")
(line_r32,) = ax_anim.plot([], [], "--m", label="r32")

# Connector 1–2
(line_c1,)  = ax_anim.plot([], [], "-.", color="orange", label="c1")

# System 3 lines
(line_r13,) = ax_anim.plot([], [], "-y", lw=2, label="r13")
(line_r23,) = ax_anim.plot([], [], "-g", label="r23")
(line_r43,) = ax_anim.plot([], [], "-b", label="r43")
(line_r33,) = ax_anim.plot([], [], "--g", label="r33")

# Bottom bracket 2–3
#(line_br23,) = ax_anim.plot([], [], ":", color="gray", label="bracket 2–3")

# Coupler 2
(line_c2,)   = ax_anim.plot([], [], "--", color="brown", label="c2")

# System 4 lines
(line_r14,) = ax_anim.plot([], [], "-k", lw=2, label="r14")
(line_r24,) = ax_anim.plot([], [], "-r", label="r24")
(line_r44,) = ax_anim.plot([], [], "-g", label="r44")
(line_r34,) = ax_anim.plot([], [], "--r", label="r34")

# Coupler 3 (optional if you have P3,P4)
(line_c3,) = ax_anim.plot([], [], "-.", color="purple", label="c3")

ax_anim.legend(loc="upper right")

# --- 9.3 Update function: sets new data for each frame ---
def update_frame(idx):
    (angles12, pts12, angles23, pts3, angles34, pts4) = all_results[idx]

    # System 1
    line_r11.set_data(
        [pts12["O2_1"][0], pts12["O4_1"][0]],
        [pts12["O2_1"][1], pts12["O4_1"][1]],
    )
    line_r21.set_data(
        [pts12["O2_1"][0], pts12["A1"][0]],
        [pts12["O2_1"][1], pts12["A1"][1]],
    )
    line_r41.set_data(
        [pts12["O4_1"][0], pts12["B1"][0]],
        [pts12["O4_1"][1], pts12["B1"][1]],
    )
    line_r31.set_data(
        [pts12["A1"][0], pts12["B1"][0]],
        [pts12["A1"][1], pts12["B1"][1]],
    )

    # System 2
    line_r12.set_data(
        [pts12["O2_2"][0], pts12["O4_2"][0]],
        [pts12["O2_2"][1], pts12["O4_2"][1]],
    )
    line_r22.set_data(
        [pts12["O2_2"][0], pts12["A2"][0]],
        [pts12["O2_2"][1], pts12["A2"][1]],
    )
    line_r42.set_data(
        [pts12["O4_2"][0], pts12["B2"][0]],
        [pts12["O4_2"][1], pts12["B2"][1]],
    )
    line_r32.set_data(
        [pts12["A2"][0], pts12["B2"][0]],
        [pts12["A2"][1], pts12["B2"][1]],
    )

    # Connector 1–2
    line_c1.set_data(
        [pts12["P1"][0], pts12["P2"][0]],
        [pts12["P1"][1], pts12["P2"][1]],
    )

    # System 3
    line_r13.set_data(
        [pts3["O2_3"][0], pts3["O4_3"][0]],
        [pts3["O2_3"][1], pts3["O4_3"][1]],
    )
    line_r23.set_data(
        [pts3["O2_3"][0], pts3["A3"][0]],
        [pts3["O2_3"][1], pts3["A3"][1]],
    )
    line_r43.set_data(
        [pts3["O4_3"][0], pts3["B3"][0]],
        [pts3["O4_3"][1], pts3["B3"][1]],
    )
    line_r33.set_data(
        [pts3["A3"][0], pts3["B3"][0]],
        [pts3["A3"][1], pts3["B3"][1]],
    )

    # bottom bracket 2–3 (O4_2 -> O2_3)
    #line_br23.set_data(
    #    [pts12["O4_2"][0], pts3["O2_3"][0]],
    #    [pts12["O4_2"][1], pts3["O2_3"][1]],
    #)

    # coupler 2 (C2L–C2R) if present
    if "C2L" in pts3 and "C2R" in pts3:
        line_c2.set_data(
            [pts3["C2L"][0], pts3["C2R"][0]],
            [pts3["C2L"][1], pts3["C2R"][1]],
        )
    else:
        line_c2.set_data([], [])

    # System 4
    line_r14.set_data(
        [pts4["O2_4"][0], pts4["O4_4"][0]],
        [pts4["O2_4"][1], pts4["O4_4"][1]],
    )
    line_r24.set_data(
        [pts4["O2_4"][0], pts4["A4"][0]],
        [pts4["O2_4"][1], pts4["A4"][1]],
    )
    line_r44.set_data(
        [pts4["O4_4"][0], pts4["B4"][0]],
        [pts4["O4_4"][1], pts4["B4"][1]],
    )
    line_r34.set_data(
        [pts4["A4"][0], pts4["B4"][0]],
        [pts4["A4"][1], pts4["B4"][1]],
    )

    # coupler 3 if you have P3,P4 in your data
    if "P3" in pts3 and "P4" in pts4:
        line_c3.set_data(
            [pts3["P3"][0], pts4["P4"][0]],
            [pts3["P3"][1], pts4["P4"][1]],
        )
    else:
        line_c3.set_data([], [])

    # IMPORTANT: do NOT change axis limits here -> fixed axes
    return

# --- 9.4 Looped animation over all configurations ---
indices = list(range(len(all_results)))
cycle_indices = itertools.cycle(indices + indices[::-1][1:-1])  # ping-pong

for idx in cycle_indices:
    if not plt.fignum_exists(fig_anim.number):
        break
    update_frame(idx)
    plt.pause(0.2)
    
# ================================================================
# 10) Compare new point cloud vs. computed configuration
# ================================================================

four_bar_linkages_new = [
    [[48.61556714979999, -6.4613681972, 51.6949069789, 3.535271727],
     [51.6949069789, 3.535271727, 61.6834871239, 3.057499096899999],
     [57.4614744999, -6.809163619299999, 48.61556714979999, -6.4613681972],
     [61.6834871239, 3.057499096899999, 57.4614744999, -6.809163619299999]],
    [[77.89278133919998, 9.130370357, 78.5861405469, -3.283368278199999],
     [78.5861405469, -3.283368278199999, 88.5467500295, -4.1700805965],
     [88.61427767030001, 8.336339949500001, 77.89278133919998, 9.130370357],
     [88.5467500295, -4.1700805965, 88.61427767030001, 8.336339949500001]],
    [[104.7931716357, -7.0168227748, 106.4790657566, 0.3450514999],
     [106.4790657566, 0.3450514999, 116.4031820584, -0.8845482837999998],
     [114.5341217729, -7.6888667091, 104.7931716357, -7.0168227748],
     [116.4031820584, -0.8845482837999998, 114.5341217729, -7.6888667091]],
    [[132.4394943465, -2.6518444088, 133.3789790866, -7.1809956861],
     [133.3789790866, -7.1809956861, 143.2622884642, -8.704214562899997],
     [142.4119775771, -3.871566397099999, 132.4394943465, -2.6518444088],
     [143.2622884642, -8.704214562899997, 142.4119775771, -3.871566397099999]]
]

coupler_links_new = [
    [118.5966231108, -0.4500182507, 121.5961888484, -0.3989752693],
    [94.3350002916, -5.201348471, 96.09023920530001, -6.160064464199999],
    [65.7038055029, 7.368966445099999, 68.46314605919999, 8.546268315499997]
]

# original r11 from first dataset (to find same link in new cloud)
r11_old = [57.4614744999, -6.8091636193,
           48.6155671498, -6.4613681972]

# ============================================================
# DEBUG: Check solved System-2 angles vs CAD (local frame)
# ============================================================

def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

# --- CAD GLOBAL angles (rad)
t12_cad_glob = math.radians(angle_deg(sys2_links[0]))
t22_cad_glob = math.radians(angle_deg(sys2_links[1]))
t32_cad_glob = math.radians(angle_deg(sys2_links[2]))
t42_cad_glob = math.radians(angle_deg(sys2_links[3]))

# --- CAD LOCAL angles (rad)
Phi2_cad = t_A1A2_initial
t12_cad_loc = wrap_pi(t12_cad_glob - Phi2_cad)
t22_cad_loc = wrap_pi(t22_cad_glob - Phi2_cad)
t32_cad_loc = wrap_pi(t32_cad_glob - Phi2_cad)
t42_cad_loc = wrap_pi(t42_cad_glob - Phi2_cad)

# --- SOLVED LOCAL angles (rad)
t12_sol = wrap_pi(angles12["t12"])
t22_sol = wrap_pi(angles12["t22"])
t32_sol = wrap_pi(angles12["t32"])   # this is already local in your model
t42_sol = wrap_pi(angles12["t42"])

# --- Print comparison
print("\n=== SYSTEM 2 ANGLE CHECK (LOCAL FRAME) ===")
print("Angle     CAD(deg)    SOLVED(deg)    ERROR(deg)")
print("------------------------------------------------")

def pr(name, cad, sol):
    print(f"{name:<6}  {math.degrees(cad):>9.4f}  {math.degrees(sol):>11.4f}  {abs(math.degrees(sol-cad)):>11.6f}")

pr("t12", t12_cad_loc, t12_sol)
pr("t22", t22_cad_loc, t22_sol)
pr("t32", t32_cad_loc, t32_sol)
pr("t42", t42_cad_loc, t42_sol)


###################################################################################################################################
# ================================================================
# 11) VERIFICATION AGAINST SECOND POINT CLOUD
#     (append this block after r11_old definition)
# ================================================================

def joints_from_ordered_links_rect(links):
    """
    Given rectangular four-bar in order [r1(bottom), r2(left), r3(top), r4(right)]
    return joint positions O2, A, O4, B as 2D numpy arrays.
    """
    r1, r2, r3, r4 = links
    O2 = np.array(r1[:2], float)   # start of bottom link
    O4 = np.array(r1[2:], float)   # end   of bottom link
    A  = np.array(r2[2:], float)   # end   of left link
    B  = np.array(r4[2:], float)   # end   of right link
    return dict(O2=O2, O4=O4, A=A, B=B)

# --- 11.1 Order new four-bar linkages (same helper as before) ---
ordered_linkages_new, idx_sorted_new = sort_mechanisms_left_to_right(
    [order_links_rectangular(l) for l in four_bar_linkages_new]
)

# also sort new couplers left->right (by midpoint x) just for plotting
cx_mid = [0.5*(c[0] + c[2]) for c in coupler_links_new]
coupler_links_new_sorted = [c for _, c in sorted(zip(cx_mid, coupler_links_new))]

print("\n=== New four-bar linkages ordered (verification set) ===\n")
for i, ordered in enumerate(ordered_linkages_new):
    print(f"New Mechanism {i+1}:")
    for j, line in enumerate(ordered, start=1):
        name = f"r{j}{i+1}_new"
        L = length(line)
        ang = angle_deg(line)
        x1, y1, x2, y2 = line
        print(
            f"  {name}: start=({x1:.3f}, {y1:.3f}), "
            f"end=({x2:.3f}, {y2:.3f}), "
            f"length={L:.3f}, angle={ang:.3f} deg"
        )
    print()

print("New coupler links (unsorted):")
for j, line in enumerate(coupler_links_new, start=1):
    name = f"c{j}_new"
    L = length(line)
    ang = angle_deg(line)
    x1, y1, x2, y2 = line
    print(
        f"  {name}: start=({x1:.3f}, {y1:.3f}), "
        f"end=({x2:.3f}, {y2:.3f}), "
        f"length={L:.3f}, angle={ang:.3f} deg"
    )

# --- 11.2 Extract measured input angle t21 from new mechanism 1 ---
# Mechanism 1 new: ordered_linkages_new[0]
# r2 is the input link O2_1 -> A1, oriented bottom->top
r21_new_line = ordered_linkages_new[0][1]
t21_meas_deg = angle_deg(r21_new_line)
t21_meas_rad = math.radians(t21_meas_deg)

print("\n=== Verification using new point cloud ===")
print(f"Measured input angle t21 from new cloud: {t21_meas_deg:.3f} deg")

# --- 11.3 Solve full mechanism for this measured angle ---
angles12_ver, _ = solve_mechanism(t21_meas_rad, prev_guess_sys2=None)
pts12_ver = compute_positions(angles12_ver)

O4_2_ver_global = pts12_ver["O4_2"]
angles23_ver, _ = solve_pair23(O4_2_ver_global, prev_guess=None)
angles34_ver, _ = solve_pair34(
    t33_val=angles23_ver["t33"],
    t23_val=angles23_ver["t23"],
    t43_val=angles23_ver["t43"],
    prev_guess=None,
)

pts3_ver = compute_positions_mech3(angles23_ver, O4_2_ver_global, br23_vec)
pts4_ver = compute_positions_mech4(angles34_ver, pts3_ver)

# --- 11.4 Build joint positions from new point cloud ---
joints_new = [joints_from_ordered_links_rect(links)
              for links in ordered_linkages_new]

# Mechanism 1 measured joints:
O2_1_meas = joints_new[0]["O2"]

# Predicted O2_1 from model:
O2_1_pred = pts12_ver["O2_1"]

# Align model to new cloud by pure translation so that O2_1 coincides
T_align = O2_1_meas - O2_1_pred

def shift(p):
    """Apply alignment translation to a 2D point."""
    return p + T_align

# --- 11.5 Compute position errors (model vs new cloud) ---
pairs = []

# mech 1: O2_1, O4_1, A1, B1
pairs += [
    (shift(pts12_ver["O2_1"]), joints_new[0]["O2"]),
    (shift(pts12_ver["O4_1"]), joints_new[0]["O4"]),
    (shift(pts12_ver["A1"]),   joints_new[0]["A"]),
    (shift(pts12_ver["B1"]),   joints_new[0]["B"]),
]

# mech 2: O2_2, O4_2, A2, B2
pairs += [
    (shift(pts12_ver["O2_2"]), joints_new[1]["O2"]),
    (shift(pts12_ver["O4_2"]), joints_new[1]["O4"]),
    (shift(pts12_ver["A2"]),   joints_new[1]["A"]),
    (shift(pts12_ver["B2"]),   joints_new[1]["B"]),
]

# mech 3: O2_3, O4_3, A3, B3
pairs += [
    (shift(pts3_ver["O2_3"]), joints_new[2]["O2"]),
    (shift(pts3_ver["O4_3"]), joints_new[2]["O4"]),
    (shift(pts3_ver["A3"]),   joints_new[2]["A"]),
    (shift(pts3_ver["B3"]),   joints_new[2]["B"]),
]

# mech 4: O2_4, O4_4, A4, B4
pairs += [
    (shift(pts4_ver["O2_4"]), joints_new[3]["O2"]),
    (shift(pts4_ver["O4_4"]), joints_new[3]["O4"]),
    (shift(pts4_ver["A4"]),   joints_new[3]["A"]),
    (shift(pts4_ver["B4"]),   joints_new[3]["B"]),
]

errors = []
for p_model, p_meas in pairs:
    diff = p_model - p_meas
    errors.append(np.linalg.norm(diff))

errors = np.array(errors)
rms_error = np.sqrt(np.mean(errors**2))
max_error = np.max(errors)

print(f"\nRMS position error (model vs new cloud): {rms_error:.6f}")
print(f"Max position error:                      {max_error:.6f}")

# --- 11.6 Plot overlay: model vs new point cloud ---
plt.figure()
plt.title("Model vs new point cloud (aligned at O2_1)")

# dummy lines for legend
plt.plot([], [], "k-", label="measured links")
plt.plot([], [], "r-", label="model links")

# measured four-bar links
for mech in ordered_linkages_new:
    for line in mech:
        x1, y1, x2, y2 = line
        plt.plot([x1, x2], [y1, y2], "k-", lw=1, alpha=0.6)

# measured couplers (sorted left->right for nicer look)
for c in coupler_links_new_sorted:
    x1, y1, x2, y2 = c
    plt.plot([x1, x2], [y1, y2], "k--", lw=1, alpha=0.6)

def plot_segment(p_start, p_end):
    plt.plot([p_start[0], p_end[0]],
             [p_start[1], p_end[1]],
             "r-", lw=2)

def plot_segment_dashed(p_start, p_end):
    plt.plot([p_start[0], p_end[0]],
             [p_start[1], p_end[1]],
             "r--", lw=2)

# System 1 model
plot_segment(shift(pts12_ver["O2_1"]), shift(pts12_ver["O4_1"]))
plot_segment(shift(pts12_ver["O2_1"]), shift(pts12_ver["A1"]))
plot_segment(shift(pts12_ver["O4_1"]), shift(pts12_ver["B1"]))
plot_segment_dashed(shift(pts12_ver["A1"]), shift(pts12_ver["B1"]))

# System 2 model
plot_segment(shift(pts12_ver["O2_2"]), shift(pts12_ver["O4_2"]))
plot_segment(shift(pts12_ver["O2_2"]), shift(pts12_ver["A2"]))
plot_segment(shift(pts12_ver["O4_2"]), shift(pts12_ver["B2"]))
plot_segment_dashed(shift(pts12_ver["A2"]), shift(pts12_ver["B2"]))

# System 3 model
plot_segment(shift(pts3_ver["O2_3"]), shift(pts3_ver["O4_3"]))
plot_segment(shift(pts3_ver["O2_3"]), shift(pts3_ver["A3"]))
plot_segment(shift(pts3_ver["O4_3"]), shift(pts3_ver["B3"]))
plot_segment_dashed(shift(pts3_ver["A3"]), shift(pts3_ver["B3"]))

# System 4 model
plot_segment(shift(pts4_ver["O2_4"]), shift(pts4_ver["O4_4"]))
plot_segment(shift(pts4_ver["O2_4"]), shift(pts4_ver["A4"]))
plot_segment(shift(pts4_ver["O4_4"]), shift(pts4_ver["B4"]))
plot_segment_dashed(shift(pts4_ver["A4"]), shift(pts4_ver["B4"]))

plt.axis("equal")
plt.grid(True, ls=":")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="best")
plt.tight_layout()
plt.show()


# TOL = 2.0  # mm

# def lines_match(l1, l2, tol):
#     p1s = np.array(l1[:2]); p1e = np.array(l1[2:])
#     p2s = np.array(l2[:2]); p2e = np.array(l2[2:])
#     if np.linalg.norm(p1s - p2s) < tol and np.linalg.norm(p1e - p2e) < tol:
#         return True
#     if np.linalg.norm(p1s - p2e) < tol and np.linalg.norm(p1e - p2s) < tol:
#         return True
#     return False

# # --- Find r11 in the NEW point cloud ---
# r11_new = None
# mech_index = None
# line_index = None

# for i, mech in enumerate(four_bar_linkages_new):
#     for j, line in enumerate(mech):
#         if lines_match(r11_old, line, TOL):
#             r11_new = np.array(line, float)
#             mech_index = i
#             line_index = j
#             break
#     if r11_new is not None:
#         break

# print("r11 found in NEW mechanism:", mech_index, "line:", line_index)
# print("r11_new =", r11_new)

# # --- Get r21 from the new mechanism geometry (ordered) ---
# ordered_new = order_links_rectangular(four_bar_linkages_new[mech_index])
# r21_new_line = ordered_new[1]  # [x1,y1,x2,y2] of side link

# # angle of r21 in radians (from new point cloud)
# t21_new = math.radians(angle_deg(r21_new_line))
# print("t21_new (deg) =", math.degrees(t21_new))

# # --- Solve mechanism for this new input angle ---
# angles12_new, _ = solve_mechanism(t21_new)
# pts12_new = compute_positions(angles12_new)

# # Also solve for mechanisms 3–4 so you have a full configuration if needed
# O4_2_new = pts12_new["O4_2"]
# angles23_new, _ = solve_pair23(O4_2_new)
# pts3_new = compute_positions_mech3(angles23_new, O4_2_new, br23_vec)
# angles34_new, _ = solve_pair34(angles23_new["t33"],
#                                angles23_new["t23"],
#                                angles23_new["t43"])
# pts4_new = compute_positions_mech4(angles34_new, pts3_new)

# # ----------------------------------------------------------------
# #  Align model configuration to the NEW point cloud via r11
# # ----------------------------------------------------------------

# # data r11 in left->right orientation
# data_r11 = orient_left_to_right(r11_new.tolist())
# x1d, y1d, x2d, y2d = data_r11
# v_d = np.array([x2d - x1d, y2d - y1d])

# # model r11 in its local frame
# model_r11 = [pts12_new["O2_1"][0], pts12_new["O2_1"][1],
#              pts12_new["O4_1"][0], pts12_new["O4_1"][1]]
# model_r11 = orient_left_to_right(model_r11)
# x1m, y1m, x2m, y2m = model_r11
# v_m = np.array([x2m - x1m, y2m - y1m])

# theta_d = math.atan2(v_d[1], v_d[0])
# theta_m = math.atan2(v_m[1], v_m[0])
# dtheta = theta_d - theta_m

# R_align = np.array([[math.cos(dtheta), -math.sin(dtheta)],
#                     [math.sin(dtheta),  math.cos(dtheta)]])

# t_align = np.array([x1d, y1d]) - R_align @ np.array([x1m, y1m])

# def transform_point(p):
#     return R_align @ p + t_align

# def transform_pts_dict(pts):
#     return {k: transform_point(np.array(v)) for k, v in pts.items()}

# pts12_new_al = transform_pts_dict(pts12_new)
# pts3_new_al  = transform_pts_dict(pts3_new)
# pts4_new_al  = transform_pts_dict(pts4_new)


# # ----------------------------------------------------------------
# #  Plot: full model (aligned) vs second point cloud
# # ----------------------------------------------------------------
# plt.figure()
# plt.title("Model prediction vs second point cloud")

# # ===== MODEL: SYSTEM 1 (aligned) =====
# plt.plot([pts12_new_al["O2_1"][0], pts12_new_al["O4_1"][0]],
#          [pts12_new_al["O2_1"][1], pts12_new_al["O4_1"][1]],
#          "-k", lw=2, label="r11 (model)")
# plt.plot([pts12_new_al["O2_1"][0], pts12_new_al["A1"][0]],
#          [pts12_new_al["O2_1"][1], pts12_new_al["A1"][1]],
#          "-b", label="r21 (model)")
# plt.plot([pts12_new_al["A1"][0], pts12_new_al["B1"][0]],
#          [pts12_new_al["A1"][1], pts12_new_al["B1"][1]],
#          "--b", label="r31 (model)")
# plt.plot([pts12_new_al["O4_1"][0], pts12_new_al["B1"][0]],
#          [pts12_new_al["O4_1"][1], pts12_new_al["B1"][1]],
#          "-g", label="r41 (model)")

# # ===== MODEL: SYSTEM 2 (aligned) =====
# plt.plot([pts12_new_al["O2_2"][0], pts12_new_al["O4_2"][0]],
#          [pts12_new_al["O2_2"][1], pts12_new_al["O4_2"][1]],
#          "-r", lw=2, label="r12 (model)")
# plt.plot([pts12_new_al["O2_2"][0], pts12_new_al["A2"][0]],
#          [pts12_new_al["O2_2"][1], pts12_new_al["A2"][1]],
#          "-m", label="r22 (model)")
# plt.plot([pts12_new_al["A2"][0], pts12_new_al["B2"][0]],
#          [pts12_new_al["A2"][1], pts12_new_al["B2"][1]],
#          "--m", label="r32 (model)")
# plt.plot([pts12_new_al["O4_2"][0], pts12_new_al["B2"][0]],
#          [pts12_new_al["O4_2"][1], pts12_new_al["B2"][1]],
#          "-c", label="r42 (model)")

# # coupler 1 (P1–P2)
# plt.plot([pts12_new_al["P1"][0], pts12_new_al["P2"][0]],
#          [pts12_new_al["P1"][1], pts12_new_al["P2"][1]],
#          "-.", color="orange", label="c1 (model)")

# # ===== MODEL: SYSTEM 3 (aligned) =====
# plt.plot([pts3_new_al["O2_3"][0], pts3_new_al["O4_3"][0]],
#          [pts3_new_al["O2_3"][1], pts3_new_al["O4_3"][1]],
#          "-y", lw=2, label="r13 (model)")
# plt.plot([pts3_new_al["O2_3"][0], pts3_new_al["A3"][0]],
#          [pts3_new_al["O2_3"][1], pts3_new_al["A3"][1]],
#          "-g", label="r23 (model)")
# plt.plot([pts3_new_al["A3"][0], pts3_new_al["B3"][0]],
#          [pts3_new_al["A3"][1], pts3_new_al["B3"][1]],
#          "--g", label="r33 (model)")
# plt.plot([pts3_new_al["O4_3"][0], pts3_new_al["B3"][0]],
#          [pts3_new_al["O4_3"][1], pts3_new_al["B3"][1]],
#          "-b", label="r43 (model)")

# # coupler 2 (C2L–C2R)
# plt.plot([pts3_new_al["C2L"][0], pts3_new_al["C2R"][0]],
#          [pts3_new_al["C2L"][1], pts3_new_al["C2R"][1]],
#          "--", color="brown", label="c2 (model)")

# # ===== MODEL: SYSTEM 4 (aligned) =====
# plt.plot([pts4_new_al["O2_4"][0], pts4_new_al["O4_4"][0]],
#          [pts4_new_al["O2_4"][1], pts4_new_al["O4_4"][1]],
#          "-k", lw=2, label="r14 (model)")
# plt.plot([pts4_new_al["O2_4"][0], pts4_new_al["A4"][0]],
#          [pts4_new_al["O2_4"][1], pts4_new_al["A4"][1]],
#          "-r", label="r24 (model)")
# plt.plot([pts4_new_al["A4"][0], pts4_new_al["B4"][0]],
#          [pts4_new_al["A4"][1], pts4_new_al["B4"][1]],
#          "--r", label="r34 (model)")
# plt.plot([pts4_new_al["O4_4"][0], pts4_new_al["B4"][0]],
#          [pts4_new_al["O4_4"][1], pts4_new_al["B4"][1]],
#          "-g", label="r44 (model)")

# # coupler 3 (P3–P4)
# plt.plot([pts3_new_al["P3"][0], pts4_new_al["P4"][0]],
#          [pts3_new_al["P3"][1], pts4_new_al["P4"][1]],
#          "-.", color="purple", label="c3 (model)")

# # ===== SECOND POINT CLOUD (measured) =====
# for mech in four_bar_linkages_new:
#     for line in mech:
#         x1, y1, x2, y2 = line
#         plt.plot([x1, x2], [y1, y2], color="0.7", lw=2)

# for line in coupler_links_new:
#     x1, y1, x2, y2 = line
#     plt.plot([x1, x2], [y1, y2], color="0.5", lw=2, linestyle="--")

# plt.axis("equal")
# plt.grid(True, ls=":")
# plt.legend(loc="upper right", fontsize=8)
# plt.tight_layout()
# plt.show()
# # # ----------------------------------------------------------------
# # #  Plot: model (aligned) vs second point cloud
# # # ----------------------------------------------------------------
# # plt.figure()
# # plt.title("Model prediction vs second point cloud")

# # # --- plot NEW point cloud in gray ---
# # for mech in four_bar_linkages_new:
# #     for line in mech:
# #         x1, y1, x2, y2 = line
# #         plt.plot([x1, x2], [y1, y2], color="0.7", lw=4)

# # for line in coupler_links_new:
# #     x1, y1, x2, y2 = line
# #     plt.plot([x1, x2], [y1, y2], color="0.5", lw=4, linestyle="--")


# # # --- model, system 1 (aligned) ---
# # plt.plot([pts12_new_al["O2_1"][0], pts12_new_al["O4_1"][0]],
# #          [pts12_new_al["O2_1"][1], pts12_new_al["O4_1"][1]],
# #          "-b", label="model r11")
# # plt.plot([pts12_new_al["O2_1"][0], pts12_new_al["A1"][0]],
# #          [pts12_new_al["O2_1"][1], pts12_new_al["A1"][1]],
# #          "-g", label="model r21")

# # # (optional) add the other links of the model as well:
# # plt.plot([pts12_new_al["O4_1"][0], pts12_new_al["B1"][0]],
# #          [pts12_new_al["O4_1"][1], pts12_new_al["B1"][1]], "-g", alpha=0.5, lw=1.5, label="model r41")
# # plt.plot([pts12_new_al["A1"][0], pts12_new_al["B1"][0]],
# #          [pts12_new_al["A1"][1], pts12_new_al["B1"][1]], "--b", alpha=0.5, lw=1.5, label="model r31")


# # plt.axis("equal")
# # plt.grid(True, ls=":")
# # plt.legend()
# # plt.tight_layout()
# # plt.show()
