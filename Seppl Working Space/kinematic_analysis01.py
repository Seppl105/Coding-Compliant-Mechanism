import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def R(th):
    return sp.Matrix([[sp.cos(th), -sp.sin(th)],
                      [sp.sin(th),  sp.cos(th)]])

def solve_two_segments(r11, r21, r31, r41,
                       r12, r22, r32, r42,
                       k1, u2, Lc,
                       T1=(0.0,0.0), phi1=0.0,
                       T2=(3.6,0.0), phi2=0.0,
                       theta21=np.deg2rad(50.0),
                       X0_deg=(-40,-80,30,-40,-60)):
    t31, t41, t22, t32, t42 = sp.symbols('t31 t41 t22 t32 t42', real=True)

    # All vector loop equations in the form R1 + R2 + ... + Rn = 0 for the real and imaginary part respectively
    eqs = []
    # Vector loop one
    eqs += [ r21*sp.cos(theta21) + r31*sp.cos(t31) - (r11 + r41*sp.cos(t41)) ]
    eqs += [ r21*sp.sin(theta21) + r31*sp.sin(t31) -  r41*sp.sin(t41) ] 
    # Vector loop two
    eqs += [ r22*sp.cos(t22)     + r32*sp.cos(t32) - (r12 + r42*sp.cos(t42)) ]
    eqs += [ r22*sp.sin(t22)     + r32*sp.sin(t32) -  r42*sp.sin(t42) ]

    
    A1_loc = sp.Matrix([r21*sp.cos(theta21), r21*sp.sin(theta21)])
    P1_loc = A1_loc + sp.Matrix([u1*r31*sp.cos(t31), k1*r31*sp.sin(t31)])

    A2_loc = sp.Matrix([r22*sp.cos(t22),     r22*sp.sin(t22)])
    P2_loc = A2_loc + sp.Matrix([u2*r32*sp.cos(t32), u2*r32*sp.sin(t32)])

    T1x,T1y = T1
    T2x,T2y = T2
    P1 = sp.Matrix([T1x, T1y]) + R(phi1) @ P1_loc
    P2 = sp.Matrix([T2x, T2y]) + R(phi2) @ P2_loc
    eqs += [ (P1-P2).dot(P1-P2) - Lc**2 ]

    F = sp.Matrix(eqs)
    X = sp.Matrix([t31,t41,t22,t32,t42])
    X0 = sp.Matrix(np.deg2rad(np.array(X0_deg, dtype=float)))

    sol = sp.nsolve(F, X, X0, tol=1e-14, maxsteps=100)
    t31v, t41v, t22v, t32v, t42v = map(float, sol)

    def joints_seg1():
        O2 = sp.Matrix([T1x, T1y])
        O4 = sp.Matrix([T1x, T1y]) + R(phi1) @ sp.Matrix([r11, 0.0])
        A  = O2 + R(phi1) @ sp.Matrix([r21*sp.cos(theta21), r21*sp.sin(theta21)])
        B  = A  + R(phi1) @ sp.Matrix([r31*sp.cos(t31v),    r31*sp.sin(t31v)])
        D  = O4 + R(phi1) @ sp.Matrix([r41*sp.cos(t41v),    r41*sp.sin(t41v)])
        P  = A  + R(phi1) @ sp.Matrix([u1*r31*sp.cos(t31v), u1*r31*sp.sin(t31v)])
        return O2, O4, A, B, D, P

    def joints_seg2():
        O2 = sp.Matrix([T2x, T2y])
        O4 = sp.Matrix([T2x, T2y]) + R(phi2) @ sp.Matrix([r12, 0.0])
        A  = O2 + R(phi2) @ sp.Matrix([r22*sp.cos(t22v),    r22*sp.sin(t22v)])
        B  = A  + R(phi2) @ sp.Matrix([r32*sp.cos(t32v),    r32*sp.sin(t32v)])
        D  = O4 + R(phi2) @ sp.Matrix([r42*sp.cos(t42v),    r42*sp.sin(t42v)])
        P  = A  + R(phi2) @ sp.Matrix([u2*r32*sp.cos(t32v), u2*r32*sp.sin(t32v)])
        return O2, O4, A, B, D, P

    O2_1, O4_1, A1, B1, D1, P1w = joints_seg1()
    O2_2, O4_2, A2, B2, D2, P2w = joints_seg2()

    out = {
        "angles_deg": {
            "t31": float(np.rad2deg(t31v)),
            "t41": float(np.rad2deg(t41v)),
            "t22": float(np.rad2deg(t22v)),
            "t32": float(np.rad2deg(t32v)),
            "t42": float(np.rad2deg(t42v)),
        },
        "seg1": {"O2":O2_1, "O4":O4_1, "A":A1, "B":B1, "D":D1, "P":P1w},
        "seg2": {"O2":O2_2, "O4":O4_2, "A":A2, "B":B2, "D":D2, "P":P2w},
        "Lc_actual": float((P1w-P2w).norm()),
        "Lc": Lc
    }
    return out

def plot_linkage(solution_dict, ax=None, colors=("tab:red","tab:blue")):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7,3))
    else:
        fig = ax.figure

    def draw_seg(seg, color, label):
        O2, O4, A, B, D, P = [np.array(seg[k], dtype=float).ravel() for k in ["O2","O4","A","B","D","P"]]
        def line(p,q,style='-'):
            ax.plot([p[0],q[0]],[p[1],q[1]], style, color=color)
        # links
        line(O2,O4,'-'); line(O2,A,'-'); line(A,B,'-'); line(O4,D,'-'); line(B,D,':')
        # joints
        xs = [O2[0],O4[0],A[0],B[0],D[0]]; ys =[O2[1],O4[1],A[1],B[1],D[1]]
        ax.scatter(xs, ys, s=25, color=color, label=label)

    draw_seg(solution_dict["seg1"], colors[0], "seg1 joints")
    draw_seg(solution_dict["seg2"], colors[1], "seg2 joints")

    P1 = np.array(solution_dict["seg1"]["P"], dtype=float).ravel()
    P2 = np.array(solution_dict["seg2"]["P"], dtype=float).ravel()
    ax.plot([P1[0],P2[0]],[P1[1],P2[1]], '-', label='green link', color='tab:green')
    ax.scatter([P1[0],P2[0]],[P1[1],P2[1]], color='tab:green', s=20)

    ax.axis('equal'); ax.grid(True, alpha=0.3)
    ax.set_title('Two coupled four-bars')
    ax.legend(loc='upper right')
    return fig, ax

# Example run with your seg-1 lengths
sol = solve_two_segments(
    1.8,2.5,2.0,2.2,   # seg-1
    1.6,1.9,2.1,2.3,   # seg-2
    u1=0.60, u2=0.35, Lc=2.0,
    T1=(0.0,0.0), phi1=np.deg2rad(0.0),
    T2=(3.6,0.0), phi2=np.deg2rad(0.0),
    theta21=np.deg2rad(50.0),
    X0_deg=(-40,-80,30,-40,-60)
)

#print("Solved angles (deg):", sol["angles_deg"])
#print("Green link length check:", f'{sol["Lc_actual"]:.3f}', "(target =", sol["Lc"], ")")

fig, ax = plot_linkage(sol)
plt.show()
