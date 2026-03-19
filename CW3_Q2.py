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
