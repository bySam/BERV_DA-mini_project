### Plot magnitude of omegas in weight matrix W.
import numpy as np
from matplotlib import pyplot as plt

def moving_average(A, N): # Moving average function from Lab2 part 2 on studium
    """A function that calculates the moving average of an array A, based on a rolling window of size N. 
    The function returns the resulting array B containing the moving average. 
    The first N-1 elements of B contains the moving average based on the assumption that the previous 10 
    elements in A had a value identical to the first element of the array, to make sure that the resulting 
    array B is of the same size as the array A."""
    A_tmp = np.ones(N-1)
    A_tmp[0:N-1] = A[0]
    A_ = np.concatenate((A_tmp, A), axis=0)
    B = np.cumsum(A_, dtype=float)
    B[N:] = B[N:] - B[:-N]
    return B[N - 1:]/N

A = np.genfromtxt('train_ts.csv', delimiter=',')
P = np.genfromtxt('samples.csv', delimiter=',')
W = np.linalg.pinv(A)@P
time = np.arange(0, 201, 1)
norms = np.zeros(201)
for i in range(201):
    norms[i] = np.linalg.norm(W[i])

# plot magnitude of omegas
plt.figure(1)
plt.plot(time, norms)

# plot magnitude of omegas starting from t = 10, smoothed with moving_average()
norms_smooth = moving_average(norms[10:], 10)
plt.figure(2)
plt.plot(time[10:], norms_smooth)

