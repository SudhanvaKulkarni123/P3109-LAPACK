import numpy as np

import matplotlib.pyplot as plt

# Load the data from the CSV file
data = np.loadtxt('/Users/sudhanvakulkarni/Documents/Precimonious-Clang-Plugins/TLAPACK-Precimonious-exp/Precimonious/GMRES_IR/scripts/e5m2_error_f_cond.csv', delimiter=',')
other_data = np.loadtxt('/Users/sudhanvakulkarni/Documents/Precimonious-Clang-Plugins/TLAPACK-Precimonious-exp/Precimonious/GMRES_IR/scripts/e5m2_error_f_cond_new.csv', delimiter=',')

# Extract the columns from the data
x = data[:, 0]
y = data[:, 1]
z = other_data[:, 0]
t = other_data[:, 1]
# Plot the values
plt.loglog(x, y)
plt.loglog(z, t)
plt.xlabel('iter')
plt.ylabel('relative back error')
plt.title('relative backwards error with and without 4th term, n = 1024, cond = 10000, block size = 32')
plt.legend(['with', 'without'])
plt.show()
