import numpy as np
from matplotlib import pyplot as plt

# set constants
mew = np.array((0.1, 1, 5, 10))
T_values = np.array((60, 60, 150, 300))
ylim_values = np.array((7, 2.5, 1.5, 1.5))
h = 10**-3

def forward_euler(X, Y, mew, h, T):
    N = int(T/h)
    # construct arrays to contain the values of x and y for each timestep
    x = np.zeros(N+1) 
    y = np.zeros(N+1)
    x[0] = X
    y[0] = Y
    # complete the timesteps by creating next value in array
    for n in range(N):
        x[n+1] = x[n] + h*(x[n] - 1/3*x[n]**3 - y[n])
        y[n+1] = y[n] + h*((1/mew) * x[n])
    return x, y

def midpoint(X, Y, mew, h, T):
    N = int(T/h)
    # construct arrays to contain the values of x and y for each timestep
    x = np.zeros(N+1)
    y = np.zeros(N+1)
    x[0] = X
    y[0] = Y
    # complete the timesteps
    for n in range(N):
        # set midpoint values
        xstar = x[n] + (h/2)*(x[n] - 1/3*x[n]**3 - y[n])
        ystar = y[n] + (h/2)*((1/mew) * x[n])
        # use midpoint values to create next value
        x[n+1] = x[n] + h*(xstar - 1/3*xstar**3 - ystar)
        y[n+1] = y[n] + h*((1/mew) * xstar)
    return x, y

# clean x values for plotting nullcline smoothly
x_null = np.linspace(-2.5, 2.5, 300)
y_null = x_null - (1/3)*x_null**3

for n in range(len(mew)):
    T = T_values[n]
    N = int(T/h)
    ylim = ylim_values[n]
    x, y = forward_euler(0.1, 0.1, mew[n], h, T)
    t = np.linspace(0, T, N+1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(t, x)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("x against time")

    plt.subplot(1, 3, 2)
    plt.plot(t, y)
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("y against time")

    plt.subplot(1, 3, 3)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-ylim, ylim)
    plt.plot(x, y, label="trajectory")
    plt.plot(x_null, y_null, linestyle="dashed", label="x-nullcline")
    plt.plot(np.array([0, 0]), np.array([-ylim, ylim]), linestyle="dashed", label="y-nullcline")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Phase plane")
    plt.legend(loc="upper right")

    plt.suptitle(f"Van der Pol equations, μ = {mew[n]}")
    plt.tight_layout()
    
    plt.show()

# equilibrium demonstration - start very close to (0,0) to show instability
mew_eq = 0.5
h_eq = 10**-3
T_eq = 30
N_eq = int(T_eq/h_eq)
x, y = forward_euler(0.01, 0.01, mew_eq, h_eq, T_eq)

x_null = np.linspace(-3, 3, 300)
y_null = x_null - (1/3)*x_null**3

plt.plot(x, y, label="trajectory")
plt.plot(x_null, y_null, linestyle="dashed", label="x-nullcline")
plt.plot(np.array([0, 0]), np.array([-3, 3]), linestyle="dashed", label="y-nullcline")
plt.plot(0, 0, "ko", markersize=8, label="equilibrium (0,0)")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Unstable equilibrium demonstration, μ = {mew_eq}")
plt.legend(loc="upper right")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.savefig("task_1_3_equilibrium.pdf", bbox_inches="tight")
plt.show()