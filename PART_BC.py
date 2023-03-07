### PART B / C

### Plot reference solutions from training_matrix 
import numpy as np
from matplotlib import pyplot as plt

Y = np.genfromtxt('train_ts.csv', delimiter=',')
P = np.genfromtxt('samples.csv', delimiter=',')

row_matrix1 = Y[0]
row_matrix2 = Y[100]
row_matrix3 = Y[1000]
row_matrix4 = Y[5000]

plt.figure(1)
plt.plot(time, row_matrix1, label='training data row 1')
plt.legend()
plt.figure(2)
plt.plot(time, row_matrix2, label='training data row 100')
plt.legend()
plt.figure(3)
plt.plot(time, row_matrix3, label='training data row 1000')
plt.legend()
plt.figure(4)
plt.plot(time, row_matrix4, label='training data row 5000')
plt.legend()

### Plot magnitude of rows in solution matrix.
def moving_average(A, N):
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

W = np.delete(W, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), axis=0)
norms = np.zeros(191)
for i in range(191):
    norms[i] = np.linalg.norm(W[i])
time = np.arange(0,191,1)
moving_average = moving_average(norms, 10)
plt.title('Magnitude of Omegas')
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.plot(time, moving_average, label='Magnitude(norm)')
plt.legend()


### MONTE-CARLO WEIGHT VERIFICATION
model = Lotkavolterra_Oscillator()

A = np.genfromtxt('train_ts.csv', delimiter=',')
P = np.genfromtxt('samples.csv', delimiter=',')
W = np.linalg.pinv(A)@P
X = 30
total = np.ones(15*X).reshape((X, 15))

for i in range(X):
    print(str((i+1)*100/X) + "% done.")
    results = model.run(algorithm = "SSA", number_of_trajectories=1)
    C = results['C']
    P = C@W
    total[i] = P
estimated_params = (np.sum(total, axis=0))/X
print(estimated_params)