import numpy as np
from matplotlib import pyplot as plt

# Set Constants
mew = 10
a = 0.8
b = 0.7
h = 0.01
T = 100
N = int(T/h)

def forward_euler(x0, y0, I):
    
    t = np.linspace(0, T, N)
    
    x = np.zeros(N)
    y = np.zeros(N)
    x[0] = x0
    y[0] = y0

    for n in range(N-1):
        x[n+1] = x[n] + h*(x[n] - 1/3*x[n]**3 - y[n] + I)
        y[n+1] = y[n] + h*((1/mew)*(x[n] - (a*y[n]) + b))

    return x, y, t
    

I_values = [0.0, 0.5, 1.0, 1.5, 2]

plt.figure(figsize=(12,8))

i = 0
for I in I_values:
    x, y, t = forward_euler(0, 0, I)

    x_vals = np.linspace(-2, 2, 400)
    x_null = x_vals - (1/3)*x_vals**3 + I
    y_null = (x_vals + b)/a

    plt.subplot(2, 3, i+1)

    plt.plot(x_vals, x_null, linestyle='dashed', label='x-nullcline')
    plt.plot(x_vals, y_null, linestyle='dashed', label='y-nullcline')
    plt.plot(x, y, label='trajectory')

    plt.title(f"I = {I}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()

    i += 1  

plt.suptitle("Phase Plane for different values of I")
plt.tight_layout()
plt.savefig("Task2PhasePlanes.png")
plt.show()
