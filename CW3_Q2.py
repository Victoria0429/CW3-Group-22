import numpy as np
import matplotlib.pyplot as plt

# define constants
mu = 10
a=0.8
b=0.7
dt=0.01

# define differential equations
def F(x, y, I):
    return x - 1/3 * x**3 - y + I
def G(x, y, mu, a, b):
    return (x - a * y + b) / mu

# create timestepping function
def forwardeuler(F, G, dt, I0):
    # start at (0,0)
    t=0
    x=0
    y=0
    
    X = [0]
    Y = [0]
    T = [0]

    # timestep for an arbitrary amount of time
    while t<100:
        dx = F(x,y,I0)
        dy = G(x,y,mu,a,b)

        y += dt * dy
        x += dt * dx

        t += dt

        X.append(x)
        Y.append(y)
        T.append(t)

    return np.array(X), np.array(Y), np.array(T)

# create array of colours, one for each line corresponding to a value of I in the time series plots, so that adjacent lines follow a gradient
colours = plt.cm.jet(np.linspace(0,1,9))

plt.figure()
# plot 9 lines for equally spaced values of I
for i in range(9):
    X, Y, T = forwardeuler(F, G, 10**-5, 0.25*i)
    plt.plot(T, X, label = f'X for I = {0.25*i:.2f}', color = colours[i])
plt.title('time series plots for (T,X)', fontsize=12)
plt.xlabel('t')
plt.ylabel('x')
# move legend to the right side of the plot to make the lines more visible
plt.legend(bbox_to_anchor=(1.4,1), loc='upper right')
plt.subplots_adjust(right = 0.75)
plt.grid()
plt.savefig('time_series_plots_(T,X).png')
plt.show()

plt.figure()
for i in range(9):
    X, Y, T = forwardeuler(F, G, 10**-5, 0.25*i)
    plt.plot(T, Y, label = f'Y for I = {0.25*i:.2f}', color = colours[i])
plt.title('time series plots for (T,Y)', fontsize=12)
plt.xlabel('t')
plt.ylabel('y')
plt.legend(bbox_to_anchor=(1.4, 1), loc='upper right')
plt.subplots_adjust(right = 0.75)
plt.grid()
plt.savefig('time_series_plots_(T,Y).png')
plt.show()

# new constants for next section
T1 = 100
N = int(T/h)

# Define time-stepping function
def forward_euler(x0, y0, I):
    
    t = np.linspace(0, T1, N)
    
    x = np.zeros(N)
    y = np.zeros(N)
    x[0] = x0
    y[0] = y0

    for n in range(N-1):
        x[n+1] = x[n] + dt*(x[n] - 1/3*x[n]**3 - y[n] + I)
        y[n+1] = y[n] + dt*((1/mu)*(x[n] - (a*y[n]) + b))

    return x, y, t
    
# Choose 5 values of I between 0 and 2
I_values = [0.0, 0.5, 1.0, 1.5, 2]

plt.figure(figsize=(12,8))

# Use for loop to make a subplot with graphs for all 5 values of I
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
    
# Update i for next iteration
    i += 1  

# Complete plot
plt.suptitle("Phase Plane for different values of I")
plt.tight_layout()
plt.savefig("Task2PhasePlanes.png")
plt.show()
