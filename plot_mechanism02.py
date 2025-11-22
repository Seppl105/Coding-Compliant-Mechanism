import math
import matplotlib.pyplot as plt

# ==== Input data ==========================================================
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
        L = length(line)
        ang = angle_deg(line)
        x1, y1, x2, y2 = line
        print(
            f"  {name}: start=({x1:.3f}, {y1:.3f}), "
            f"end=({x2:.3f}, {y2:.3f}), "
            f"length={L:.3f}, angle={ang:.3f} deg"
        )
    print()

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
