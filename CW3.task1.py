import numpy as np
from matplotlib import pyplot as plt

# set constants 
mew = 0.5
h = 10**-5
T = 20
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


x, y = forward_euler(1,1)
X, Y = midpoint(1,1)

# plot results, using dashed line for one to distingush when lines are over the top of eachother
plt.plot(x, y, label = "Forward Euler method")
plt.plot(X, Y, linestyle = "dashed", label = "Midpoint method")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Time Stepping of Van der Pol equations with μ = 0.5, T = 20, h = 0.0001 ")
plt.legend()
plt.show()

T = 10

# time-stepping the van der Pol equation using the midpoint method
def midpoint(X, Y, h, N):
    x = np.zeros(N+1)
    y = np.zeros(N+1)
    x[0] = X
    y[0] = Y
    
    for n in range(N):
        # calculate the midpoint
        xstar = x[n] + (h/2)*(x[n] - 1/3*x[n]**3 - y[n])
        ystar = y[n] + (h/2)*(mew**-1 * x[n])
        
        # take the full step using the derivatives at the midpoint
        x[n+1] = x[n] + h*(xstar - 1/3*xstar**3 - ystar)
        y[n+1] = y[n] + h*(mew**-1 * xstar)
        
    return x, y

# run the reference calculation with a very small step size
h_ref = 10**-5
N_ref = int(T / h_ref)
x_ref, y_ref = midpoint(1, 1, h_ref, N_ref)

# grab the final x value at t=T to use as our exact reference
x_reference = x_ref[N_ref]

# test different step sizes to see how the error scales
h_values = [10**-2, 5*10**-2, 10**-1, 2*10**-1]
errors = []

for h in h_values:
    N = int(T / h)  # making sure we land exactly on t=T
    x, y = midpoint(1, 1, h, N)
    
    # error calc: absolute difference from the reference value at t=T
    errors.append(abs(x[N] - x_reference))

h_array = np.array(h_values)

# create a theoretical O(h^2) line to compare against
reference_line = (errors[0] / h_values[0]**2) * h_array**2

# plot
plt.loglog(h_values, errors, 'bo-', label='Measured error')
plt.loglog(h_array, reference_line, 'r--', label=r'$O(h^2)$ reference')
plt.xlabel("h (step size)")
plt.ylabel(r"$\varepsilon(h) = |x(T) - x_{ref}|$")
plt.title("Convergence of midpoint method with μ = 0.5, T = 10")
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.show()
