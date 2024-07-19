import matplotlib.pyplot as plt
import numpy as np


data = {'FP32-FP8GEMM-GMRES':(4,8), 'FP32-FP16GEMM-GMRES':(0,8), 'FP32-BFP8GEMM-GMRES':(4,6), 
    'FP32-FP8GEMM-FGMRES' : (8,8), 'FP32-FP16GEMM-FGMRES':(8,8), 'FP32-BFP8GEMM-FGMRES':(12,12)}

courses = list(data.keys())
values = np.array(list(data.values()))

fig = plt.figure(figsize = (10, 5))

# creating the stacked bar plot
plt.bar(courses, values[:, 0], label='fp64 iters')

plt.bar(courses, values[:, 1], bottom=values[:, 0], label='fp32 iters')
plt.legend()
plt.show()
