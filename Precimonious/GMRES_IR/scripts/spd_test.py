import numpy as np
import matplotlib.pyplot as plt

# Function to generate a Symmetric Positive Definite (SPD) matrix with a given condition number
def generate_spd_matrix(n, cond_number):
    # Create a random orthogonal matrix Q using QR decomposition
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    
    # Generate eigenvalues that spread across a given condition number
    eigenvalues = np.linspace(1.0/cond_number, 1.0, n)
    
    # Create the SPD matrix as Q * D * Q^T, where D is the diagonal matrix of eigenvalues
    D = np.diag(eigenvalues)
    spd_matrix = Q @ D @ Q.T
    
    return spd_matrix

# Function to plot the sorted diagonal of a matrix
def plot_sorted_diagonal(matrix):
    diagonal = np.diag(matrix)
    sorted_diagonal = np.sort(diagonal)
    
    plt.plot(sorted_diagonal, marker='o')
    plt.title("Sorted Diagonal Elements of SPD Matrix")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

# Parameters
n = 10  # Matrix size
cond_number = 1000  # Desired condition number

# Generate SPD matrix
spd_matrix = generate_spd_matrix(n, cond_number)

# Plot the sorted diagonal of the matrix
plot_sorted_diagonal(spd_matrix)
