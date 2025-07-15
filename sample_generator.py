import numpy as np
import torch 
import denoising
import importlib
denoising = importlib.reload(denoising)
import utils
utils = importlib.reload(utils)


# import packages
import math
np.math = math  # Redirect numpy.math to the built-in math module

import glob

# Define patch number
patch = "3"

base_path = f"/obs/atsouros/ScatteringDenoising/planck_data/BK_CMB_S4_north_patch_v4/"

# Find the file using a wildcard
i_path = glob.glob(f"{base_path}signal/patch_{patch}/patch_{patch}_I*.npy")[0]
q_path = glob.glob(f"{base_path}signal/patch_{patch}/patch_{patch}_Q*.npy")[0]
u_path = glob.glob(f"{base_path}signal/patch_{patch}/patch_{patch}_U*.npy")[0]

def downsample(image):
    func = utils.downsample_by_four
    return func(image)

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
nuisance_Q = sorted(glob.glob(f"{base_path}nuisance/patch_{patch}/Stokes_Q/patch_{patch}_noise_Q*.npy"))
nuisance_U = sorted(glob.glob(f"{base_path}nuisance/patch_{patch}/Stokes_U/patch_{patch}_noise_U*.npy"))

# Load and downsample
contamination_arr_Q = np.stack([downsample(np.load(f))[None, :, :] for f in nuisance_Q], axis=0)
contamination_arr_U = np.stack([downsample(np.load(f))[None, :, :] for f in nuisance_U], axis=0)

# Stack into shape (N_maps, 3, 768, 768)
contamination_arr = np.stack([contamination_arr_Q, contamination_arr_U], axis=1)

image_target = np.load(f"/obs/atsouros/ScatteringDenoising/image_denoised_patch_{patch}.npy")
image_target = (image_target[0][None, :, :], image_target[1][None, :, :])

def add_white_noise(image_target, snr_db):
    image_init = []
    for target in image_target:
        signal_power = np.mean(target**2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(scale=np.sqrt(noise_power), size=target.shape)
        image_init.append(target + noise)
    return tuple(image_init)

threshold_func = None
remove_edge = True
n_realizations = 10
denoised_images = []

for i in range(n_realizations):
    print(f'Starting generation of sample {i}')
    image_init = add_white_noise(image_target, snr_db=2)
    running_map = denoising.denoise(image_target, fixed_img=signal_I, seed=i, print_each_step=True,
                                    steps=25, n_batch=25, s_cov_func=threshold_func, image_init=image_init,
                                    remove_edge=remove_edge, precision='double', if_large_batch=False,
                                    epochNo=i, mode='synthesis')
    image_syn_Q, image_syn_U = running_map[0], running_map[1]
    image_denoised = np.stack([image_syn_Q[0], image_syn_U[0]])  # shape: (2, H, W)
    denoised_images.append(image_denoised)

# Stack all results into shape (10, 2, H, W)
denoised_images = np.stack(denoised_images, axis=0)

# Save results
np.save(f"samples_{patch}.npy", denoised_images)
