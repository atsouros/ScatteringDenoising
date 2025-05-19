import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

# import packages
import math
import sys
# sys.path.append('.')  # or full path to CompSep
np.math = math  # Redirect numpy.math to the built-in math module

import denoising
import importlib
denoising = importlib.reload(denoising)
import utils
utils = importlib.reload(utils)

contamination_arr = np.load('contamination_arr.npy')
data = np.load('data.npy')

image_target = data
std = denoising.compute_std(image_target, contamination_arr = contamination_arr, precision='double')
std_double = denoising.compute_std_double(image_target, contamination_arr = contamination_arr, precision='double')

image_init = image_target

n_epochs = 3 #number of epochs
# decontaminate
for i in range(n_epochs):
    print(f'Starting epoch {i+1}')
    running_map = denoising.denoise(image_target, contamination_arr = contamination_arr, std = std, std_double=std_double ,seed=0, print_each_step=True, steps = 25, n_batch = 25, image_init = image_init, precision='double')
    running_map = (running_map[0], running_map[1])

    std = denoising.compute_std(running_map, contamination_arr = contamination_arr)
    std_double = denoising.compute_std_double(running_map, contamination_arr = contamination_arr)

image_syn_Q = running_map[0]
image_syn_U = running_map[1]

# Convert tuples to NumPy arrays
data = np.stack([data[0], data[1]])  # Shape: (2, ...)
image_denoised = np.stack([image_syn_Q[0], image_syn_U[0]])  # Ensure it's an array

results = np.array([data, image_denoised])
np.save(f"recon.npy", results)