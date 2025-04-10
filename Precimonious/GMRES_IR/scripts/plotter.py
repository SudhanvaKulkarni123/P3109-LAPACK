import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive plotting

import re
import numpy as np
import matplotlib.pyplot as plt
import os

def parse_flops(filename):
    # Updated to handle multiple eps_prime sections
    flops_data = {}  # Dictionary to store flops data for each eps_prime
    current_eps_prime = None
    current_method = None

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue

            # Match eps_prime lines
            eps_match = re.match(r'eps_prime\s*:\s*([0-9.eE+-]+)', line)
            if eps_match:
                current_eps_prime = float(eps_match.group(1))
                print(current_eps_prime)
                flops_data[current_eps_prime] = {
                    'Cholesky': {}, 
                    'CG': {}, 
                    'vanilla_Cholesky': {}, 
                }
                current_method = None
                continue  # Move to the next line

            if 'vanilla Cholesky flops' in line:
                current_method = 'vanilla_Cholesky'
                continue  # Move to the next line
            elif 'CG flops' in line:
                current_method = 'CG'
                continue  # Move to the next line
            elif 'Cholesky flops' in line:
                current_method = 'Cholesky'
                continue


            # Match lines like 'double flops : 1.08373e+09'
            match = re.match(r'(\w+) flops\s*:\s*([0-9.eE+-]+)', line)
            if match and current_eps_prime is not None and current_method is not None:
                flop_type = match.group(1).lower()  # Ensure consistent key naming
                flop_value = float(match.group(2))
                flops_data[current_eps_prime][current_method][flop_type] = flop_value

    return flops_data

def compute_total_work(method_flops, alpha):
    # Compute total work for a given method and alpha
    double_flops = method_flops.get('double', 0)
    float_flops = method_flops.get('float', 0)
    half_flops = method_flops.get('half', 0)
    bfloat_flops = method_flops.get('bfloat', 0)
    fp8_flops = method_flops.get('fp8', 0)

    total_work = (4 * alpha * double_flops +
                  2 * alpha * float_flops +
                  alpha * half_flops +
                  alpha * bfloat_flops +
                  fp8_flops)
    return total_work

def plot_ratio(flops_data, alphas, filename):
    plt.figure(figsize=(8,6))

    for eps_prime in sorted(flops_data.keys()):
        ratios = []
        for alpha in alphas:
            method_flops = flops_data[eps_prime]

            # Compute total work for each component
            cholesky_work = compute_total_work(method_flops['Cholesky'], alpha)
            cg_work = compute_total_work(method_flops['CG'], alpha)
            vanilla_cholesky_work = compute_total_work(method_flops['vanilla_Cholesky'], alpha)




            # Compute min(vanilla Cholesky, vanilla CG)
            min_vanilla_work = min(vanilla_cholesky_work, cg_work)

            
            # Compute the ratio
            denominator = cholesky_work
            ratio = min_vanilla_work / denominator if denominator != 0 else np.nan
            ratios.append(ratio)

        plt.plot(alphas, ratios, marker='o', label=f'eps_prime = {eps_prime}')

    plt.xlabel('Alpha')
    plt.ylabel('(min(vanilla Cholesky, vanilla CG) / (preconditioned CG + Cholesky))')
    plt.title('Ratio of Total Work (weighted FLOPs) vs Alpha for Different eps_prime Values')
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    os.makedirs('results', exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    plot_filename = f"results/plot-{base_filename}.png"
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")

    plt.close()  # Close the figure to free memory

# Example usage
if __name__ == '__main__':
    print("Current working directory:", os.getcwd())

    filename = 'results/n1024cond1000.txt'  # Replace with your actual filename
    flops_data = parse_flops(filename)
    alphas = np.linspace(1, 5, 40)  # Adjust the range and number of alpha values as needed
    plot_ratio(flops_data, alphas, filename)
