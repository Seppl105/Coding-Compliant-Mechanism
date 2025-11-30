import math
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1) INPUT: your four-bar link point clouds (red rectangles)
# ==========================================================
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

# ==========================================================
# 2) Helpers for geometry + ordering rectangular 4-bars
# ==========================================================
def length(line):
    x1, y1, x2, y2 = line
    return math.hypot(x2 - x1, y2 - y1)

def angle(line):
    x1, y1, x2, y2 = line
    return math.atan2(y2 - y1, x2 - x1)

def order_links_rectangular(linkage):
    """
    Put the 4 segments of a rectangular four-bar into order:
      r1: bottom  (ground)  left→right
      r2: left side         bottom→top
      r3: top               left→right
      r4: right side        bottom→top
    """
    lines = [list(l) for l in linkage]
    mids = [((l[0] + l[2]) * 0.5, (l[1] + l[3]) * 0.5) for l in lines]
    ys = [m[1] for m in mids]
    min_y, max_y = min(ys), max(ys)
    tol = 0.1 * (max_y - min_y + 1e-9)

    bottom_idx = [i for i, (x, y) in enumerate(mids) if y <= min_y + tol]
    top_idx    = [i for i, (x, y) in enumerate(mids) if y >= max_y - tol]

    def horiz_score(i):
        a = angle(lines[i])
        # closeness to horizontal
        return abs(((a + math.pi/2) % math.pi) - math.pi/2)

    r1_idx = min(bottom_idx, key=horiz_score)
    r3_idx = min(top_idx,    key=horiz_score)

    remaining = [i for i in range(4) if i not in (r1_idx, r3_idx)]
    xs_rem = [mids[i][0] for i in remaining]
    left_idx  = remaining[xs_rem.index(min(xs_rem))]
    right_idx = remaining[xs_rem.index(max(xs_rem))]

    def orient_lr(line):
        x1, y1, x2, y2 = line
        return line if x2 >= x1 else [x2, y2, x1, y1]

    def orient_bt(line):
        x1, y1, x2, y2 = line
        return line if y2 >= y1 else [x2, y2, x1, y1]

    r1 = orient_lr(lines[r1_idx])
    r3 = orient_lr(lines[r3_idx])
    r2 = orient_bt(lines[left_idx])
    r4 = orient_bt(lines[right_idx])
    return [r1, r2, r3, r4]

def joints_from_ordered(links):
    """
    Given [r1, r2, r3, r4] return the four joints:
        O2 (bottom-left), A (top-left), O4 (bottom-right), B (top-right)
    """
    r1, r2, r3, r4 = links
    O2 = np.array(r1[:2], float)
    O4 = np.array(r1[2:], float)
    A  = np.array(r2[2:], float)
    B  = np.array(r4[2:], float)
    return O2, A, O4, B

# ==========================================================
# 3) Build a local 4-bar model for EACH mechanism
# ==========================================================
ordered_linkages = [order_links_rectangular(l) for l in four_bar_linkages]

mechs = []   # list of dicts, one per four-bar
for idx, links in enumerate(ordered_linkages, start=1):
    r1, r2, r3, r4 = links
    O2, A, O4, B = joints_from_ordered(links)

    d = length(r1)      # ground
    a = length(r2)      # input
    b = length(r3)      # coupler
    c = length(r4)      # output

    theta1 = angle(r1)                  # ground orientation (global)
    theta2_0 = angle([*O2, *A])         # initial input angle in global frame

    mechs.append(dict(
        idx=idx,
        O2_0=O2,
        O4_0=O4,
        A_0=A,
        B_0=B,
        a=a, b=b, c=c, d=d,
        theta1=theta1,
        theta2_0=theta2_0,
    ))

# use mechanism 1 input as reference
theta2_1_0 = mechs[0]["theta2_0"]

# ==========================================================
# 4) Generic geometric 4-bar solver (local frame)
# ==========================================================
def solve_fourbar(a, b, c, d, theta2_local, open_branch=True):
    """
    Standard 4-bar in local frame:
      O2=(0,0), O4=(d,0)
      a: input  (O2-A)
      b: coupler (A-B)
      c: output (O4-B)

    Returns local joints O2,A,B,O4 and angles theta3, theta4 (local).
    """
    O2_loc = np.array([0.0, 0.0])
    O4_loc = np.array([d, 0.0])

    A = O2_loc + a * np.array([math.cos(theta2_local),
                               math.sin(theta2_local)])

    # intersection of two circles
    dx = O4_loc[0] - A[0]
    dy = O4_loc[1] - A[1]
    D = math.hypot(dx, dy)

    if D > a + b + c:   # crude feasibility check
        raise ValueError("4-bar cannot close for this input angle")

    x = (b**2 - c**2 + D**2) / (2 * D)
    h_sq = b**2 - x**2
    h = math.sqrt(max(h_sq, 0.0))

    ex, ey = dx / D, dy / D
    Pm = A + x * np.array([ex, ey])

    if open_branch:
        B = Pm + h * np.array([-ey, ex])
    else:
        B = Pm - h * np.array([-ey, ex])

    theta3 = math.atan2(B[1] - A[1], B[0] - A[0])
    theta4 = math.atan2(B[1] - O4_loc[1], B[0] - O4_loc[0])

    return dict(O2=O2_loc, A=A, B=B, O4=O4_loc,
                theta3=theta3, theta4=theta4)

def local_to_global(p_local, O2_global, theta1):
    R = np.array([[math.cos(theta1), -math.sin(theta1)],
                  [math.sin(theta1),  math.cos(theta1)]])
    return O2_global + R @ p_local

# ==========================================================
# 5) Given t21, solve ALL four-bars
# ==========================================================
def solve_all_mechs(t21_rad, open_branch=True):
    """
    t21_rad : global input angle for mechanism 1 (r2_1).
    All other mechanisms have their input link rotated by the
    same delta from their own CAD reference pose.
    """
    delta = t21_rad - theta2_1_0  # change from reference

    configs = []
    for m in mechs:
        a, b, c, d = m["a"], m["b"], m["c"], m["d"]
        theta1     = m["theta1"]
        theta2_0   = m["theta2_0"]

        theta2_global = theta2_0 + delta
        theta2_local  = theta2_global - theta1

        sol_loc = solve_fourbar(a, b, c, d, theta2_local,
                                open_branch=open_branch)

        O2_g = local_to_global(sol_loc["O2"], m["O2_0"], theta1)
        A_g  = local_to_global(sol_loc["A"],  m["O2_0"], theta1)
        B_g  = local_to_global(sol_loc["B"],  m["O2_0"], theta1)
        O4_g = local_to_global(sol_loc["O4"], m["O2_0"], theta1)

        configs.append(dict(
            idx=m["idx"],
            O2=O2_g, A=A_g, B=B_g, O4=O4_g,
            theta2=theta2_global,
            theta3=sol_loc["theta3"] + theta1,
            theta4=sol_loc["theta4"] + theta1,
        ))
    return configs

# ==========================================================
# 6) Example: sweep t21 and plot all mechanisms
# ==========================================================
t21_deg_list = np.linspace(60, 120, 7)      # degrees
all_sets = [solve_all_mechs(math.radians(a)) for a in t21_deg_list]

plt.figure()
plt.title("All four four-bar mechanisms driven by t21")
colors = ["C0", "C1", "C2", "C3"]  # one color per mechanism

for configs, t_deg in zip(all_sets, t21_deg_list):
    alpha = 0.25 + 0.75 * (t_deg - t21_deg_list[0]) / (t21_deg_list[-1] - t21_deg_list[0])

    for cfg in configs:
        i = cfg["idx"] - 1
        col = colors[i]

        O2, A, B, O4 = cfg["O2"], cfg["A"], cfg["B"], cfg["O4"]

        # ground
        plt.plot([O2[0], O4[0]], [O2[1], O4[1]], "-", color=col, lw=3, alpha=alpha)
        # input
        plt.plot([O2[0], A[0]], [O2[1], A[1]], "-", color=col, lw=2, alpha=alpha)
        # output
        plt.plot([O4[0], B[0]], [O4[1], B[1]], "-", color=col, lw=2, alpha=alpha)
        # coupler
        plt.plot([A[0], B[0]], [A[1], B[1]], "--", color=col, lw=1.5, alpha=alpha)

        # joints
        plt.scatter([O2[0], A[0], B[0], O4[0]],
                    [O2[1], A[1], B[1], O4[1]],
                    s=10, color=col, alpha=alpha)

plt.axis("equal")
plt.grid(True, ls=":")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
