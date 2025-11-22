import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

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

ordered_linkages = [order_links_rectangular(l) for l in four_bar_linkages]

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

# ================================================================
# 7) SOLVE AND PLOT MULTIPLE CONFIGURATIONS
# ================================================================

# --- choose input angles you want to see (in degrees) ---
t21_degs = [70.0, 80.0 , 90.0, 100.0, 109.473, 120.0, 130.0]   # you can change this list
t21_list = [math.radians(d) for d in t21_degs]

all_results = []
guess_sys2 = None

for t21_val in t21_list:
    angles, guess_sys2 = solve_mechanism(t21_val, guess_sys2)
    pts = compute_positions(angles)

    # simple sanity check for connector length
    conn_len = np.linalg.norm(pts["P2"] - pts["P1"])
    print(f"t21 = {math.degrees(t21_val):7.3f} deg |P2-P1| = {conn_len:.6f} (target l_c = {l_c1:.6f})")

    all_results.append((angles, pts))

# ================================================================
# 8) PLOTS
# ================================================================

# --- Plot a single reference configuration (e.g. the middle one) ---
ref_angles, ref_pts = all_results[len(all_results) // 2]

plt.figure()
plt.title("Reference configuration")
# System 1
plt.plot([ref_pts["O2_1"][0], ref_pts["O4_1"][0]],
         [ref_pts["O2_1"][1], ref_pts["O4_1"][1]], "-k", lw=2, label="r11")
plt.plot([ref_pts["O2_1"][0], ref_pts["A1"][0]],
         [ref_pts["O2_1"][1], ref_pts["A1"][1]], "-b", label="r21")
plt.plot([ref_pts["O4_1"][0], ref_pts["B1"][0]],
         [ref_pts["O4_1"][1], ref_pts["B1"][1]], "-g", label="r41")
plt.plot([ref_pts["A1"][0], ref_pts["B1"][0]],
         [ref_pts["A1"][1], ref_pts["B1"][1]], "--b", label="r31")

# System 2
plt.plot([ref_pts["O2_2"][0], ref_pts["O4_2"][0]],
         [ref_pts["O2_2"][1], ref_pts["O4_2"][1]], "-r", lw=2, label="r12")
plt.plot([ref_pts["O2_2"][0], ref_pts["A2"][0]],
         [ref_pts["O2_2"][1], ref_pts["A2"][1]], "-m", label="r22")
plt.plot([ref_pts["O4_2"][0], ref_pts["B2"][0]],
         [ref_pts["O4_2"][1], ref_pts["B2"][1]], "-c", label="r42")
plt.plot([ref_pts["A2"][0], ref_pts["B2"][0]],
         [ref_pts["A2"][1], ref_pts["B2"][1]], "--m", label="r32")

# Connector
plt.plot([ref_pts["P1"][0], ref_pts["P2"][0]],
         [ref_pts["P1"][1], ref_pts["P2"][1]], "-.", color="orange", label="connector")
plt.scatter([ref_pts["P1"][0], ref_pts["P2"][0]],
            [ref_pts["P1"][1], ref_pts["P2"][1]], s=20, color="orange")

plt.axis("equal")
plt.grid(True, ls=":")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()

# --- Plot all configurations in one figure ---
plt.figure()
plt.title("Multiple configurations of both four-bars")

for idx, (angles, pts) in enumerate(all_results):
    alpha = 0.25 + 0.75 * idx / max(1, len(all_results) - 1)

    # System 1
    plt.plot([pts["O2_1"][0], pts["O4_1"][0]],
             [pts["O2_1"][1], pts["O4_1"][1]], "-k", lw=1, alpha=alpha)
    plt.plot([pts["O2_1"][0], pts["A1"][0]],
             [pts["O2_1"][1], pts["A1"][1]], "-b", alpha=alpha)
    plt.plot([pts["O4_1"][0], pts["B1"][0]],
             [pts["O4_1"][1], pts["B1"][1]], "-g", alpha=alpha)
    plt.plot([pts["A1"][0], pts["B1"][0]],
             [pts["A1"][1], pts["B1"][1]], "--b", alpha=alpha)

    # System 2
    plt.plot([pts["O2_2"][0], pts["O4_2"][0]],
             [pts["O2_2"][1], pts["O4_2"][1]], "-r", lw=1, alpha=alpha)
    plt.plot([pts["O2_2"][0], pts["A2"][0]],
             [pts["O2_2"][1], pts["A2"][1]], "-m", alpha=alpha)
    plt.plot([pts["O4_2"][0], pts["B2"][0]],
             [pts["O4_2"][1], pts["B2"][1]], "-c", alpha=alpha)
    plt.plot([pts["A2"][0], pts["B2"][0]],
             [pts["A2"][1], pts["B2"][1]], "--m", alpha=alpha)

    # Connector
    plt.plot([pts["P1"][0], pts["P2"][0]],
             [pts["P1"][1], pts["P2"][1]], "-.", color="orange", alpha=alpha)

plt.axis("equal")
plt.grid(True, ls=":")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()

plt.show()
