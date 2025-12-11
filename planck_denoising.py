import numpy as np
import torch

import denoising
import utils
import sys

# import packages
import math
np.math = math  # Redirect numpy.math to the built-in math module

import glob

# Define patch number

patch = "129"

base_path = f"/obs/atsouros/ScatteringDenoising/planck_data/BK_CMB_S4_north_patch_v4/"

# # Find the file using a wildcard
i_path = glob.glob(f"{base_path}signal/patch_{patch}/patch_{patch}_I857*.npy")[0]
q_path = glob.glob(f"{base_path}signal/patch_{patch}/patch_{patch}_Q353*.npy")[0]
u_path = glob.glob(f"{base_path}signal/patch_{patch}/patch_{patch}_U353*.npy")[0]

# i_path = glob.glob(f"{base_path}signal/patch_{patch}/patch_{patch}_I857_NoPre*.npy")[0]
# q_path = glob.glob(f"{base_path}signal/patch_{patch}/patch_{patch}_Q353_NoPre*.npy")[0]
# u_path = glob.glob(f"{base_path}signal/patch_{patch}/patch_{patch}_U353_NoPre*.npy")[0]

def downsample(image):
    func = utils.downsample_by_four
    return func(image)

# # Load it
signal_Q = np.load(q_path)
signal_Q = downsample(signal_Q)
signal_Q = signal_Q[None, :, :]

# signal_Q = np.rot90(signal_Q, k=1, axes=(1, 2)).copy()


# Load it
signal_U = np.load(u_path)
signal_U = downsample(signal_U)
signal_U = signal_U[None, :, :]

# signal_U = np.rot90(signal_U, k=1, axes=(1, 2)).copy()


# Load it
signal_I = np.load(i_path)
signal_I = downsample(signal_I)
signal_I = signal_I[None, :, :]

# signal_I = np.rot90(signal_U, k=1, axes=(1, 2)).copy()


# Define base nuisance directory

# Get sorted list of file paths from Stokes_Q and Stokes_U directories
nuisance_Q = sorted(glob.glob(f"{base_path}nuisance/patch_{patch}/Stokes_Q/patch_{patch}_noise_Q353*.npy"))
nuisance_U = sorted(glob.glob(f"{base_path}nuisance/patch_{patch}/Stokes_U/patch_{patch}_noise_U353*.npy"))

# Load and downsample
contamination_arr_Q = np.stack([downsample(np.load(f))[None, :, :] for f in nuisance_Q], axis=0)
contamination_arr_U = np.stack([downsample(np.load(f))[None, :, :] for f in nuisance_U], axis=0)

# Stack into shape (N_maps, 3, 768, 768)
contamination_arr = np.stack([contamination_arr_Q, contamination_arr_U], axis=1)


# --- Optional: downscale the clean signal and add one nuisance map ---

if patch == '129':

    # # Downscale the clean signals
    signal_Q = signal_Q / 10.0
    signal_U = signal_U / 10.0

    # Add one chosen nuisance map to the downscaled signals
    chosen_idx = 0
    signal_Q = signal_Q + contamination_arr_Q[chosen_idx]
    signal_U = signal_U + contamination_arr_U[chosen_idx]

    # Remove the chosen nuisance map from both Q and U
    contamination_arr_Q = np.delete(contamination_arr_Q, chosen_idx, axis=0)
    contamination_arr_U = np.delete(contamination_arr_U, chosen_idx, axis=0)

    # Restack to keep contamination_arr consistent
    contamination_arr = np.stack([contamination_arr_Q, contamination_arr_U], axis=1)

########## optional part ends

image_target = (signal_Q, signal_U)
threshold_func = None
remove_edge = False

std = {
    'single': denoising.compute_std(image_target, contamination_arr=contamination_arr,
                                    s_cov_func=threshold_func, remove_edge=remove_edge, precision='double'),

    'partial': denoising.compute_std_partial(image_target, contamination_arr, signal_I,
                                           remove_edge=remove_edge, precision='double'),                               

    'double': denoising.compute_std_double(image_target, contamination_arr=contamination_arr,
                                           remove_edge=remove_edge, precision='double'),

    'noise_mean_std': denoising.noise_mean_std(contamination_arr, remove_edge=remove_edge, precision='double')
}

image_init = image_target

n_epochs = 4 #number of epochs
# decontaminate
for i in range(n_epochs):
    print(f'Starting epoch {i+1}')
    running_map = denoising.denoise(image_target, contamination_arr = contamination_arr, std = std, fixed_img=signal_I, seed=0, print_each_step=True, 
                                    steps = 45, n_batch = 25, s_cov_func=threshold_func, image_init = image_init, remove_edge=remove_edge, precision='double', 
                                    if_large_batch=False, epochNo = i)
    running_map = (running_map[0], running_map[1])
    image_init = running_map
    torch.cuda.empty_cache()

    if (i + 1) % 2 == 0:

        std = {
            'single': denoising.compute_std(running_map, contamination_arr=contamination_arr,
                                            s_cov_func=threshold_func, remove_edge=remove_edge, precision='double'),

            'partial': denoising.compute_std_partial(running_map, contamination_arr, signal_I,
                                                remove_edge=remove_edge, precision='double'),                               

            'double': denoising.compute_std_double(running_map, contamination_arr=contamination_arr,
                                                remove_edge=remove_edge, precision='double'),

            'noise_mean_std': std['noise_mean_std']
            }
        # image_init = image_target
        
    # np.save(f"image_denoised_patch_{patch}_iter={i}.npy",  np.stack([running_map[0][0], running_map[1][0]]))

image_syn_Q = running_map[0]
image_syn_U = running_map[1]

# Convert tuples to NumPy arrays
image_denoised = np.stack([image_syn_Q[0], image_syn_U[0]])  # Ensure it's an array

# Save results
if chosen_idx is not None:
    np.save(f"patch_{patch}_chosenidx={chosen_idx}.npy", image_denoised)
else:
    np.save(f"patch_{patch}_remove_edge={remove_edge}.npy", image_denoised)
