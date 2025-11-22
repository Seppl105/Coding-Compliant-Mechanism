import matplotlib.pyplot as plt
import math

# ==== Input data ==========================================================
# Four-bar linkages (nested array):
# Format: four_bar_linkages[linkage_index][line_index] = [x_start, y_start, x_end, y_end]
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

# Coupler links:
# Format: coupler_links[line_index] = [x_start, y_start, x_end, y_end]
coupler_links = [
    [55.8698004946, 9.5132738931, 58.86793206260001, 9.4074102978],
    [85.61870658990001, -3.2093366617, 87.61608151709999, -3.1068993637],
    [105.8473830683, 5.26720582, 108.7778450626, 4.6250278854],
]

# ==== Helper function: length and angle ===================================

def link_properties(x1, y1, x2, y2):
    """Return length and angle (deg) of link from (x1, y1) to (x2, y2)."""
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)              # sqrt(dx^2 + dy^2)
    angle_rad = math.atan2(dy, dx)           # angle from +x (CCW), in radians
    angle_deg = math.degrees(angle_rad)      # convert to degrees
    return length, angle_deg

# ==== Compute and print properties ========================================

print("=== Four-bar linkages ===")
for i, linkage in enumerate(four_bar_linkages):
    print(f"\nFour-bar {i+1}:")
    for j, line in enumerate(linkage):
        x1, y1, x2, y2 = line
        length, angle_deg = link_properties(x1, y1, x2, y2)
        print(
            f"  Link {j+1}: "
            f"start=({x1:.3f}, {y1:.3f}), end=({x2:.3f}, {y2:.3f}), "
            f"length={length:.3f}, angle={angle_deg:.3f} deg"
        )

print("\n=== Coupler links ===")
for k, line in enumerate(coupler_links):
    x1, y1, x2, y2 = line
    length, angle_deg = link_properties(x1, y1, x2, y2)
    print(
        f"  Coupler {k+1}: "
        f"start=({x1:.3f}, {y1:.3f}), end=({x2:.3f}, {y2:.3f}), "
        f"length={length:.3f}, angle={angle_deg:.3f} deg"
    )

# ==== Plotting code =======================================================

fig, ax = plt.subplots(figsize=(8, 4))

# Colors for different linkages (cycled automatically if you add more)
linkage_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# Plot four-bar linkages
for i, linkage in enumerate(four_bar_linkages):
    color = linkage_colors[i % len(linkage_colors)]
    for j, line in enumerate(linkage):
        x1, y1, x2, y2 = line
        lbl = f"Four-bar {i+1}" if j == 0 else None
        ax.plot([x1, x2], [y1, y2],
                marker="o", linestyle="-", color=color, label=lbl)

        # (Optional) annotate angle near the middle of the link
        length, angle_deg = link_properties(x1, y1, x2, y2)
        xm = 0.5 * (x1 + x2)
        ym = 0.5 * (y1 + y2)
        ax.text(xm, ym, f"{angle_deg:.1f}°", fontsize=7, ha="center", va="center")

# Plot coupler links
for k, line in enumerate(coupler_links):
    x1, y1, x2, y2 = line
    lbl = "Coupler links" if k == 0 else None
    ax.plot([x1, x2], [y1, y2],
            marker="s", linestyle="--", color="black", label=lbl)

    # Annotate coupler angles as well
    _, angle_deg = link_properties(x1, y1, x2, y2)
    xm = 0.5 * (x1 + x2)
    ym = 0.5 * (y1 + y2)
    ax.text(xm, ym, f"{angle_deg:.1f}°", fontsize=7, ha="center", va="center")

# Make it look nice
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Four-Bar Mechanisms with Coupler Links\n(link lengths & angles printed in console)")
ax.set_aspect("equal", adjustable="box")
ax.grid(True, linestyle=":", linewidth=0.5)
ax.legend()

plt.tight_layout()
plt.show()