import numpy as np
import torch

import denoising
import utils

# import packages
import math
np.math = math  # Redirect numpy.math to the built-in math module

import glob

# Define patch number
patch = "166"


base_path = f"/obs/atsouros/ScatteringDenoising/planck_data/BK_CMB_S4_north_patch/"

# Find the file using a wildcard
q_path = glob.glob(f"{base_path}signal/patch_{patch}/patch_{patch}_Q*.npy")[0]
u_path = glob.glob(f"{base_path}signal/patch_{patch}/patch_{patch}_U*.npy")[0]

# Load it
signal_Q = np.load(q_path)
signal_Q = utils.downsample_by_four(signal_Q)
signal_Q = signal_Q[None, :, :]

# Load it
signal_U = np.load(u_path)
signal_U = utils.downsample_by_four(signal_U)
signal_U = signal_U[None, :, :]

# Define base nuisance directory

# Get sorted list of file paths from Stokes_Q and Stokes_U directories
nuisance_Q = sorted(glob.glob(f"{base_path}nuisance/patch_{patch}/Stokes_Q/patch_{patch}_noise_Q*.npy"))
nuisance_U = sorted(glob.glob(f"{base_path}nuisance/patch_{patch}/Stokes_U/patch_{patch}_noise_U*.npy"))

# Load and downsample
contamination_arr_Q = np.stack([utils.downsample_by_four(np.load(f))[None, :, :] for f in nuisance_Q], axis=0)
contamination_arr_U = np.stack([utils.downsample_by_four(np.load(f))[None, :, :] for f in nuisance_U], axis=0)

# Stack into shape (N_maps, 2, 768, 768)
contamination_arr = np.stack([contamination_arr_Q, contamination_arr_U], axis=1)

image_target = (signal_Q, signal_U)
threshold_func = None
remove_edge = False
std = denoising.compute_std(image_target, contamination_arr = contamination_arr, s_cov_func = threshold_func, remove_edge=remove_edge, precision='double')
std_double = denoising.compute_std_double(image_target, contamination_arr = contamination_arr, remove_edge=remove_edge, precision='double')

image_init = image_target

n_epochs = 3 #number of epochs
# decontaminate
for i in range(n_epochs):
    print(f'Starting epoch {i+1}')
    running_map = denoising.denoise(image_target, contamination_arr = contamination_arr, std = std, std_double=std_double ,seed=0, print_each_step=True, steps = 25, n_batch = 25, s_cov_func=threshold_func, image_init = image_init, remove_edge=remove_edge, precision='double')
    running_map = (running_map[0], running_map[1])

    std = denoising.compute_std(running_map, contamination_arr = contamination_arr, remove_edge=remove_edge)
    std_double = denoising.compute_std_double(running_map, contamination_arr = contamination_arr, remove_edge=remove_edge)

image_syn_Q = running_map[0]
image_syn_U = running_map[1]

# Convert tuples to NumPy arrays
data = np.stack([signal_Q[0], signal_U[0]])  # Shape: (2, ...)
image_denoised = np.stack([image_syn_Q[0], image_syn_U[0]])  # Ensure it's an array

# Save results
np.save(f"data_patch_{patch}.npy", data)
np.save(f"image_denoised_patch_{patch}.npy", image_denoised)

# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True

# # import packages
# import math
# import sys
# # sys.path.append('.')  # or full path to CompSep
# np.math = math  # Redirect numpy.math to the built-in math module

# import denoising
# import importlib
# denoising = importlib.reload(denoising)
# import utils
# utils = importlib.reload(utils)

# contamination_arr = np.load('contamination_arr.npy')
# data = np.load('data.npy')

# image_target = (data[0], data[1])
# std = denoising.compute_std(image_target, contamination_arr = contamination_arr, precision='double')
# std_double = denoising.compute_std_double(image_target, contamination_arr = contamination_arr, precision='double')

# image_init = image_target

# n_epochs = 3 #number of epochs
# # decontaminate
# for i in range(n_epochs):
#     print(f'Starting epoch {i+1}')
#     running_map = denoising.denoise(image_target, contamination_arr = contamination_arr, std = std, std_double=std_double ,seed=0, print_each_step=True, steps = 25, n_batch = 25, image_init = image_init, precision$
#     running_map = (running_map[0], running_map[1])

#     std = denoising.compute_std(running_map, contamination_arr = contamination_arr)
#     std_double = denoising.compute_std_double(running_map, contamination_arr = contamination_arr)

# image_syn_Q = running_map[0]
# image_syn_U = running_map[1]

# # Convert tuples to NumPy arrays
# data = np.stack([data[0], data[1]])  # Shape: (2, ...)
# image_denoised = np.stack([image_syn_Q[0], image_syn_U[0]])  # Ensure it's an array

# np.save("data.npy", data)
# np.save("image_denoised.npy", image_denoised)