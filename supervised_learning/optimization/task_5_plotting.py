import numpy as np
import matplotlib.pyplot as plt

# Define the function and its gradient
def f(x):
    return x**2

def grad(x):
    return 2*x

# Parameters
alpha = 0.1  # Learning rate
beta1 = 0.9  # Momentum weight
x_sgd = 10.0  # Initial value for standard GD
x_momentum = 10.0  # Initial value for momentum GD
v = 0.0  # Initial momentum term
steps = 50  # Number of steps

# Store the paths
path_sgd = [x_sgd]
path_momentum = [x_momentum]

# Optimization loop
for _ in range(steps):
    # Standard Gradient Descent
    grad_sgd = grad(x_sgd)
    x_sgd = x_sgd - alpha * grad_sgd
    path_sgd.append(x_sgd)

    # Gradient Descent with Momentum
    grad_momentum = grad(x_momentum)
    v = beta1 * v + (1 - beta1) * grad_momentum
    x_momentum = x_momentum - alpha * v
    path_momentum.append(x_momentum)

# Plot the results
plt.plot(path_sgd, label="Standard GD", linestyle="--")
plt.plot(path_momentum, label="GD with Momentum", linestyle="-")
plt.xlabel("Steps")
plt.ylabel("x")
plt.title("Gradient Descent vs Gradient Descent with Momentum")
plt.legend()
plt.grid()
plt.show()