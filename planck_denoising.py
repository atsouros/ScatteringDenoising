import numpy as np

import denoising
import importlib
import utils


# import packages
import math
np.math = math  # Redirect numpy.math to the built-in math module

import glob

# Define patch number
patch = "166"

base_path = f"/Users/tsouros/Desktop/Planck data/BK_CMB_S4_north_patch/"

# Find the file using a wildcard
i_path = glob.glob(f"{base_path}signal/patch_{patch}/patch_{patch}_I*.npy")[0]
q_path = glob.glob(f"{base_path}signal/patch_{patch}/patch_{patch}_Q*.npy")[0]
u_path = glob.glob(f"{base_path}signal/patch_{patch}/patch_{patch}_U*.npy")[0]

def downsample(image):
    func = utils.downsample_by_four
    return func(func(image))

# # Load it
signal_Q = np.load(q_path)
signal_Q = downsample(signal_Q)
signal_Q = signal_Q[None, :, :]

# Load it
signal_U = np.load(u_path)
signal_U = downsample(signal_U)
signal_U = signal_U[None, :, :]

# Load it
signal_I = np.load(i_path)
signal_I = downsample(signal_I)
signal_I = signal_I[None, :, :]

# Define base nuisance directory

# Get sorted list of file paths from Stokes_Q and Stokes_U directories
nuisance_I = sorted(glob.glob(f"{base_path}nuisance/patch_{patch}/Stokes_I/patch_{patch}_noise_I*.npy"))
nuisance_Q = sorted(glob.glob(f"{base_path}nuisance/patch_{patch}/Stokes_Q/patch_{patch}_noise_Q*.npy"))
nuisance_U = sorted(glob.glob(f"{base_path}nuisance/patch_{patch}/Stokes_U/patch_{patch}_noise_U*.npy"))

# Load and downsample
contamination_arr_I = np.stack([downsample(np.load(f))[None, :, :] for f in nuisance_I], axis=0)
contamination_arr_Q = np.stack([downsample(np.load(f))[None, :, :] for f in nuisance_Q], axis=0)
contamination_arr_U = np.stack([downsample(np.load(f))[None, :, :] for f in nuisance_U], axis=0)

# Stack into shape (N_maps, 3, 768, 768)
contamination_arr = np.stack([contamination_arr_I, contamination_arr_Q, contamination_arr_U], axis=1)

image_target = (signal_I, signal_Q, signal_U)
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
    running_map = (running_map[0], running_map[1], running_map[2])

    std = denoising.compute_std(running_map, contamination_arr = contamination_arr, remove_edge=remove_edge)
    std_double = denoising.compute_std_double(running_map, contamination_arr = contamination_arr, remove_edge=remove_edge)

image_syn_I = running_map[0]
image_syn_Q = running_map[1]
image_syn_U = running_map[2]

# Convert tuples to NumPy arrays
image_denoised = np.stack([image_syn_I[0], image_syn_Q[0], image_syn_U[0]])  # Ensure it's an array

# Save results
np.save(f"image_denoised_patch_{patch}.npy", image_denoised)