import numpy as np
from matplotlib import pyplot as plt

# set constants 
mew = 4
h = 10**-5
T = 50
N = int(T/h)


def forward_euler(X, Y):
    
    # construct arrays to contain the values of x and y for each timestep
    x = np.zeros(N+1) 
    y = np.zeros(N+1)
    x[0] = X
    y[0] = Y


    # complete the timesteps by creating next value in array
    for n in range(N):

        x[n+1] = x[n] + h*(x[n] - 1/3*x[n]**3 - y[n])
        y[n+1] = y[n] + h*(mew**-1 * x[n])


    return x, y



def midpoint(X, Y):
    # construct arrays to contain the values of x and y for each timestep
    x = np.zeros(N+1)
    y = np.zeros(N+1)
    x[0] = X
    y[0] = Y

    # complete the timesteps
    for n in range(N):

        # set midpoint values
        xstar = x[n] + (h/2)*(x[n] - 1/3*x[n]**3 - y[n])
        ystar = y[n] + (h/2)*(mew**-1 * x[n])

        # use midpoint values to create next value
        x[n+1] = x[n] + h*(xstar - 1/3*xstar**3 - ystar)
        y[n+1] = y[n] + h*(mew**-1 * xstar)

    return x, y


x, y = forward_euler(0.00001,0.00001)
X, Y = midpoint(0.00001,0.00001)

# plot results, using dashed line for one to distingush when lines are over the top of eachother
plt.plot(x, y, label = "Forward Euler method")
plt.plot(X, Y, linestyle = "dashed", label = "Midpoint method")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Time Stepping of Van der Pol equations with μ = 0.5, T = 20, h = 0.0001 ")
plt.legend()
plt.show()
