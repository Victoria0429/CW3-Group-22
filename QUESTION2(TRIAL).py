"""
FitzHugh-Nagumo (Q2) — Complete Solution
MTH1003 CW3

Parameters: mu=10, a=0.8, b=0.7, varying I in [0, 2]
Time-stepping: Forward Euler
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================
mu = 10.0
a = 0.8
b = 0.7
h = 1e-4          # time step
T = 100.0          # total time (long enough for transients to die)
N = int(T / h)

# initial conditions
x0, y0 = 0.5, 0.5

# ============================================================
# FORWARD EULER TIME-STEPPER
# ============================================================
def fhn_euler(I_val, x0, y0, h, N):
    """Forward Euler for FitzHugh-Nagumo system."""
    x = np.zeros(N + 1)
    y = np.zeros(N + 1)
    x[0], y[0] = x0, y0
    for n in range(N):
        x[n+1] = x[n] + h * (x[n] - x[n]**3 / 3 - y[n] + I_val)
        y[n+1] = y[n] + h * (1/mu) * (x[n] - a * y[n] + b)
    return x, y

# ============================================================
# EQUILIBRIUM AND EIGENVALUE COMPUTATION
# ============================================================
def find_equilibrium(I_val):
    """Solve the cubic x^3 + 3(1/a - 1)x + 3b/a - 3I = 0 numerically."""
    coeffs = [1, 0, 3*(1/a - 1), 3*b/a - 3*I_val]
    roots = np.roots(coeffs)
    # take the real root (cubic always has at least one)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    xeq = real_roots[0]
    yeq = (xeq + b) / a
    return xeq, yeq

def compute_eigenvalues(xeq):
    """Eigenvalues of the Jacobian at equilibrium."""
    # Jacobian: [[1 - xeq^2, -1], [1/mu, -a/mu]]
    # Characteristic equation: lambda^2 - (1 - xeq^2 - a/mu)*lambda + (a(xeq^2 - 1) + 1)/mu = 0
    trace = (1 - xeq**2) + (-a / mu)
    det = (1 - xeq**2) * (-a / mu) - (-1) * (1 / mu)
    disc = trace**2 - 4 * det
    if disc >= 0:
        l1 = (trace + np.sqrt(disc)) / 2
        l2 = (trace - np.sqrt(disc)) / 2
    else:
        l1 = (trace + 1j * np.sqrt(-disc)) / 2
        l2 = (trace - 1j * np.sqrt(-disc)) / 2
    return l1, l2

def classify_equilibrium(l1, l2):
    """Classify based on eigenvalues."""
    re1, re2 = np.real(l1), np.real(l2)
    if np.abs(np.imag(l1)) > 1e-10:  # complex
        if re1 > 0:
            return "unstable spiral"
        elif re1 < 0:
            return "stable spiral"
        else:
            return "centre"
    else:  # real
        if re1 > 0 and re2 > 0:
            return "unstable node"
        elif re1 < 0 and re2 < 0:
            return "stable node"
        else:
            return "saddle point"


# ============================================================
# FIGURE 1: TIME SERIES FOR RANGE OF I VALUES (like existing Fig 7)
# ============================================================
I_values_ts = np.linspace(0, 2, 9)
cmap = plt.cm.coolwarm

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('FitzHugh–Nagumo: time series for varying $I$', fontsize=14)

for i, I_val in enumerate(I_values_ts):
    colour = cmap(i / (len(I_values_ts) - 1))
    x, y = fhn_euler(I_val, x0, y0, h, N)
    t = np.linspace(0, T, N + 1)
    ax1.plot(t, x, color=colour, label=f'$I = {I_val:.2f}$', linewidth=0.8)
    ax2.plot(t, y, color=colour, label=f'$I = {I_val:.2f}$', linewidth=0.8)

ax1.set_xlabel('$t$')
ax1.set_ylabel('$x(t)$')
ax1.set_title('$x$ against time')
ax1.legend(fontsize=7, loc='upper left')

ax2.set_xlabel('$t$')
ax2.set_ylabel('$y(t)$')
ax2.set_title('$y$ against time')
ax2.legend(fontsize=7, loc='upper left')

plt.tight_layout()
plt.savefig('fhn_time_series.png', dpi=200, bbox_inches='tight')
plt.show()


# ============================================================
# FIGURE 2: PHASE PLANES FOR MULTIPLE I VALUES
# ============================================================
I_values_pp = [0.0, 0.5, 1.0, 1.5]

# nullcline data (same for all I for y-nullcline, shifts for x-nullcline)
x_null = np.linspace(-3, 3, 500)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('FitzHugh–Nagumo: phase planes for varying $I$', fontsize=14)

for idx, I_val in enumerate(I_values_pp):
    ax = axes[idx // 2, idx % 2]

    # --- time-step a trajectory ---
    x, y = fhn_euler(I_val, x0, y0, h, N)

    # --- equilibrium ---
    xeq, yeq = find_equilibrium(I_val)
    l1, l2 = compute_eigenvalues(xeq)
    classification = classify_equilibrium(l1, l2)

    # --- nullclines ---
    y_xnull = x_null - x_null**3 / 3 + I_val     # x-nullcline: y = x - x^3/3 + I
    y_ynull = (x_null + b) / a                     # y-nullcline: y = (x + b) / a

    # --- plot trajectory ---
    ax.plot(x, y, 'b-', linewidth=0.8, label='trajectory')

    # --- plot nullclines (Q1 style: orange dashed, green dashed) ---
    ax.plot(x_null, y_xnull, '--', color='tab:orange', linewidth=1.2, label='$x$-nullcline')
    ax.plot(x_null, y_ynull, '--', color='tab:green', linewidth=1.2, label='$y$-nullcline')

    # --- plot equilibrium ---
    ax.plot(xeq, yeq, 'ko', markersize=6, label=f'equilibrium ({classification})')

    # --- formatting ---
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.5, 3.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(f'$I = {I_val}$')
    ax.legend(fontsize=7, loc='upper left')

plt.tight_layout()
plt.savefig('fhn_phase_planes.png', dpi=200, bbox_inches='tight')
plt.show()


# ============================================================
# FIGURE 3: INDIVIDUAL PHASE PLANES WITH TIME SERIES (Q1 STYLE)
# ============================================================
# This mirrors Q1's approach: for each I, show x(t), y(t), and phase plane side by side.

I_values_detail = [0.0, 0.5, 1.0, 1.5]

for I_val in I_values_detail:
    x, y = fhn_euler(I_val, x0, y0, h, N)
    t = np.linspace(0, T, N + 1)

    xeq, yeq = find_equilibrium(I_val)
    l1, l2 = compute_eigenvalues(xeq)
    classification = classify_equilibrium(l1, l2)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f'FitzHugh–Nagumo, $I = {I_val}$', fontsize=13)

    # x(t)
    ax1.plot(t, x, 'b-', linewidth=0.6)
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$x(t)$')
    ax1.set_title('$x$ against time')

    # y(t)
    ax2.plot(t, y, 'b-', linewidth=0.6)
    ax2.set_xlabel('$t$')
    ax2.set_ylabel('$y(t)$')
    ax2.set_title('$y$ against time')

    # phase plane
    y_xnull = x_null - x_null**3 / 3 + I_val
    y_ynull = (x_null + b) / a

    ax3.plot(x, y, 'b-', linewidth=0.8, label='trajectory')
    ax3.plot(x_null, y_xnull, '--', color='tab:orange', linewidth=1.2, label='$x$-nullcline')
    ax3.plot(x_null, y_ynull, '--', color='tab:green', linewidth=1.2, label='$y$-nullcline')
    ax3.plot(xeq, yeq, 'ko', markersize=6, label=f'equilibrium')
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-1.5, 3.5)
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$y$')
    ax3.set_title('Phase plane')
    ax3.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f'fhn_detail_I{I_val:.1f}.png', dpi=200, bbox_inches='tight')
    plt.show()

    print(f"I = {I_val:.2f}:  x_eq = {xeq:.4f},  y_eq = {yeq:.4f}")
    print(f"  Eigenvalues: {l1:.4f}, {l2:.4f}")
    print(f"  Classification: {classification}")
    print()


# ============================================================
# TABLE: EIGENVALUE CLASSIFICATION SUMMARY
# ============================================================
print("=" * 70)
print(f"{'I':>6}  {'x_eq':>10}  {'y_eq':>10}  {'lambda_1':>20}  {'lambda_2':>20}  {'type'}")
print("=" * 70)

I_table = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
for I_val in I_table:
    xeq, yeq = find_equilibrium(I_val)
    l1, l2 = compute_eigenvalues(xeq)
    classification = classify_equilibrium(l1, l2)

    if np.abs(np.imag(l1)) > 1e-10:
        l1_str = f"{np.real(l1):.4f} ± {np.abs(np.imag(l1)):.4f}i"
        l2_str = ""
    else:
        l1_str = f"{np.real(l1):.4f}"
        l2_str = f"{np.real(l2):.4f}"

    print(f"{I_val:>6.2f}  {xeq:>10.4f}  {yeq:>10.4f}  {l1_str:>20}  {l2_str:>20}  {classification}")

print("=" * 70)
