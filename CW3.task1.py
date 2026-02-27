import numpy as np
from matplotlib import pyplot as plt

mew = 0.5
h = 0.1
T = 50
N = int(T/h)


def forward_euler(X, Y):
    x = np.zeros(N+1)
    y = np.zeros(N+1)
    x[0] = X
    y[0] = Y

    for n in range(N):

        x[n+1] = x[n] + h*(x[n] - 1/3*x[n]**3 - y[n])
        y[n+1] = y[n] + h*(mew**-1 * x[n])


    return x, y



def midpoint(X, Y):
    x = np.zeros(N+1)
    y = np.zeros(N+1)
    x[0] = X
    y[0] = Y

    for n in range(N):

        xstar = x[n] + (h/2)*(x[n] - 1/3*x[n]**3 - y[n])
        ystar = y[n] + (h/2)*(mew**-1 * x[n])

        x[n+1] = x[n] + h*(xstar - 1/3*xstar**3 - ystar)
        y[n+1] = y[n] + h*(mew**-1 * xstar)

    return x, y


x, y = forward_euler(1,1)
X, Y = midpoint(1,1)

plt.plot(X, Y)
plt.plot(x, y)
plt.show()
