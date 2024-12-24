"""import torch

# Create a tensor with requires_grad=True
x = torch.tensor([1.0], requires_grad=True)

# Define the functions
g = x**2 - 1
h = torch.cos(g)
n = h * h
m = torch.log(n)

# Perform backpropagation
m.backward()
with torch.no_grad():
    manual_x=-4*x*torch.sin(x**2-1)/torch.cos(x**2-1)
# Print the gradient of x
print(x.grad)

print(manual_x)"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

# Define the signals based on the given expressions
n = np.arange(8)
x_n = np.exp(-1j * 3 * np.pi * n / 4)  # x[n] for n = 0, ..., 7
y_n = np.exp(1j * 2 * np.pi * n / 3)   # y[n] for n = 0, ..., 7

# Part (a): Length-8 DFT of x[n]
X_k = fft(x_n, 8)
magnitude_X_k = np.abs(X_k)

# Part (b): Length-8 DFT of y[n]
Y_k = fft(y_n, 8)
magnitude_Y_k = np.abs(Y_k)

# Part (c): Zero-padded x[n] to length 10 and compute length-10 DFT
v_n = np.pad(x_n, (0, 2), 'constant')  # zero-pad to length 10
V_k = fft(v_n, 10)
magnitude_V_k = np.abs(V_k)

# Plotting results
plt.figure(figsize=(15, 10))

# Part (a) Plot
plt.subplot(3, 1, 1)
plt.stem(np.arange(8), magnitude_X_k)
plt.title("Magnitude of Length-8 DFT of x[n]")
plt.xlabel("Frequency Bin")
plt.ylabel("|X(k)|")
plt.grid(True)

# Part (b) Plot
plt.subplot(3, 1, 2)
plt.stem(np.arange(8), magnitude_Y_k)
plt.title("Magnitude of Length-8 DFT of y[n]")
plt.xlabel("Frequency Bin")
plt.ylabel("|Y(k)|")
plt.grid(True)

# Part (c) Plot
plt.subplot(3, 1, 3)
plt.stem(np.arange(10), magnitude_V_k)
plt.title("Magnitude of Length-10 DFT of v[n] (Zero-padded x[n])")
plt.xlabel("Frequency Bin")
plt.ylabel("|V(k)|")
plt.grid(True)

plt.tight_layout()
plt.show()