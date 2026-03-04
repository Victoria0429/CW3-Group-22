import numpy as np
from matplotlib import pyplot as plt

# set constants 
mew = np.array((0.1, 1, 5, 10))
h = 10**-5
T = 20
N = int(T/h)


def forward_euler(X, Y, mew):
    
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
        ystar = y[n] + (h/2)*((1/mew) * x[n])

        # use midpoint values to create next value
        x[n+1] = x[n] + h*(xstar - 1/3*xstar**3 - ystar)
        y[n+1] = y[n] + h*((1/mew) * xstar)

    return x, y

for n in range(len(mew)) :
    x, y = forward_euler(0.1, 0.1, mew[n])
    t = np.linspace(0, T, N+1)
    
    plt.subplot(1,3,1)
    plt.plot(t, x)

    plt.subplot(1,3,2)
    plt.plot(t, y)
             
    plt.subplot(1,3,3)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5,2.5)
    plt.plot(x, y)
    plt.plot(x, (x-1/3*x**3), linestyle = "dashed", label = "X-nullcline")
    plt.plot(np.array([0,0]), np.array([-3,3]), linestyle = "dashed", label = "Y-nullcline")
    plt.legend()

    plt.show()

    
