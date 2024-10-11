import numpy as np
from scipy.sparse.linalg import cg
from scipy.linalg import norm

# Generate a random dense 1024x1024 matrix
A = np.random.rand(1024, 1024)

# Make the matrix symmetric and positive definite
A = A @ A.T

# Generate a random right-hand side vector
b = np.random.rand(1024)

# Set a convergence tolerance
tolerance = 1e-16

# Solve using Conjugate Gradient method
x, info = cg(A, b, tol=tolerance, maxiter=10)

# Print the number of iterations if converged, or max iterations if it didn't
if info > 0:
    print(f"CG converged in {info} iterations.")
elif info == 0:
    print(f"CG converged to the desired tolerance in less than {10} iterations.")
else:
    print("CG did not converge within the maximum number of iterations.")

