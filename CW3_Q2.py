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
#plt.show()

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
#plt.show()


# create (x,y) phase plane plotting function
def streamplot(x_range, y_range, I):
    x = np.arange(*x_range)
    y = np.arange(*y_range)

    # numerically find solution for x_eq to plot the x - nullcline
    roots = np.roots((1, 0, 3/4, 21/8 - 3*I))
    x_eq = roots[np.isreal(roots)].real[0]
    
    X1, Y1 = np.meshgrid(x, y)

    xdot = X1 - 1/3 * X1**3 - Y1 + I
    ydot = 1/10 * (X1 - 0.8 * Y1 + 0.7)

    plt.figure()
    plt.title(f'(x,y) phaseplane for I = {I}')
    plt.streamplot(X1, Y1, xdot, ydot, density = 1.3)
    # plot y - nullcline
    plt.plot(x, (10/8) * x + 7/8, color='green', linewidth=1, linestyle='--', label = f'\u1E8F = 0')
    # plot range of value of equilibrium points
    plt.plot((0,2),(7/8, 10/8 * 2 + 7/8), color = 'red', linewidth = 1.5, label = 'range of values of equilibrium points')
    plt.xlim(x_range[0],x_range[1])
    plt.ylim(y_range[0],y_range[1])
    # plot x - nullcline
    plt.axvline(x = x_eq, color='orange', linewidth=1, linestyle='--', label = '\u1E8B = 0')
    # plot equilibrium point
    plt.plot(x_eq, 10/8 * x_eq + 7/8, color = 'black', marker = 'o', label = 'equilibrium point')
    # move legend below the graph
    plt.legend(bbox_to_anchor=(0.5,-0.5), loc='lower center')
    plt.subplots_adjust(bottom = 0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.savefig('xy_phaseplane.png')
    plt.show()
    return None

streamplot((-6,6,0.01),(-6,6,0.01),1)

# the cubic polynomial function for which x_eq is a root is monotonic increasing, and I appears only in a constant term of negative magnitude, hence as I increases, x_eq increases, and the range of values of the equilibrium point correspond to a section of the y - nullcline
