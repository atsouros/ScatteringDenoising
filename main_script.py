# import packages
import numpy as np
import math
import sys
np.math = math  # Redirect numpy.math to the built-in math module

import denoising
import utils
import time

# Load the column density map from file
logN_H = np.load('Archive/Turb_3.npy')[1]  
# logN_H = utils.downsample_by_four(logN_H)
nx, ny = logN_H.shape

# Define constant dust temperature
T_d = 10  # Typical Planck dust temperature in K

# int_val = []
# for nu in nu_list:
#     int_val.append(np.mean(modified_blackbody(logN_H, T_d, nu*1e9)))
# int_val = np.array(int_val)

# Observation frequency (e.g., 353 GHz in Hz)

# Compute the mock observed intensity map in μK_CMB
nu = (217,353)
I_nu_map_μK_nu1 = utils.modified_blackbody(logN_H, T_d, nu[0]*1e9)
I_nu_map_μK_nu2 = utils.modified_blackbody(logN_H, T_d, nu[1]*1e9)
I_nu_map_μK = (I_nu_map_μK_nu1, I_nu_map_μK_nu2)

# Ensure inputs have shape (1, H, W)
dust_nu1 = I_nu_map_μK_nu1[None, :, :]
dust_nu2 = I_nu_map_μK_nu2[None, :, :]

dust = (dust_nu1, dust_nu2)

# --- Parameters ---
n_realizations = 100
SNR = 1
amplitude = 2.
spectral_index = -1.7

nx, ny = dust_nu1.shape[-2], dust_nu1.shape[-1]

# --- Compute noise variances ---
variance_nu1 = (np.std(dust_nu1) / SNR) ** 2
variance_nu2 = (np.std(dust_nu2) / SNR) ** 2
variance = (variance_nu1, variance_nu2)

# --- Create contamination_arr with shape (n_realizations, 2, 1, H, W) ---
contamination_arr = np.zeros((n_realizations, 2, 1, nx, ny), dtype=np.float32)

for i in range(n_realizations):
    # Shared CMB: shape (1, H, W)
    cmb_map = utils.generate_cmb_map(n_x=nx, n_y=ny, amplitude=amplitude, spectral_index=spectral_index)
    cmb_map = cmb_map.cpu().numpy()[None, :, :]

    # Independent noise: shape (1, H, W)
    noise_nu1 = np.random.normal(0, np.sqrt(variance_nu1), (1, nx, ny))
    noise_nu2 = np.random.normal(0, np.sqrt(variance_nu2), (1, nx, ny))

    # Total contamination: shape (1, H, W)
    contamination_arr[i, 0] = noise_nu1 + cmb_map
    contamination_arr[i, 1] = noise_nu2 + cmb_map

contamination_arr_nu1 = contamination_arr[:, 0]  # shape: (Mn, 1, H, W)
contamination_arr_nu2 = contamination_arr[:, 1]  # shape: (Mn, 1, H, W)

noise_nu1 = np.random.normal(0, np.sqrt(variance_nu1), dust_nu1.shape)
noise_nu2 = np.random.normal(0, np.sqrt(variance_nu2), dust_nu2.shape)

noise = (noise_nu1, noise_nu2)

cmb_map = utils.generate_cmb_map(n_x=nx, n_y=ny, amplitude=amplitude, spectral_index=spectral_index)
cmb_map = cmb_map.cpu().numpy()[None, :, :]

data_nu1 = dust_nu1 + noise_nu1 + cmb_map
data_nu2 = dust_nu2 + noise_nu2 + cmb_map

# define target
data = (data_nu1, data_nu2)

# target is 
image_target = data

# definte initial maps for optimisation
image_init = image_target

thresholding = False
if thresholding:
    M, N, J, L = dust_nu1.shape[-2], dust_nu1.shape[-1], 7, 4
    st_calc = denoising.Scattering2d(M, N, J, L)
    st_calc.add_ref(ref=data_nu1)
    s_cov = st_calc.scattering_cov(data_nu1, use_ref=True, 
                        normalization='P00', pseudo_coef=1
                    )
    st_calc.add_ref_ab(ref_a=data_nu1, ref_b=data_nu2)
    s_cov_2fields = st_calc.scattering_cov_2fields(data_nu1, data_nu2, use_ref=True, 
                        normalization='P00'
                    )
    threshold_func = denoising.threshold_func(s_cov)
    threshold_func_2fields = denoising.threshold_func(s_cov_2fields, two_fields=True)

else:
    threshold_func = None

std = denoising.compute_std(image_target, contamination_arr = contamination_arr, s_cov_func = threshold_func)
std_double = denoising.compute_std_double(image_target, contamination_arr = contamination_arr, s_cov_func = threshold_func_2fields)

n_epochs = 3 #number of epochs
# decontaminate
for i in range(n_epochs):
    print(f'Starting epoch {i+1}')
    running_map = denoising.denoise_double(image_target, contamination_arr = contamination_arr, std = std, std_double=std_double, seed=0, print_each_step=True, steps = 25, n_batch = 25, s_cov_func=threshold_func, s_cov_func_2fields=threshold_func_2fields)

    std = denoising.compute_std(running_map, contamination_arr = contamination_arr, s_cov_func = threshold_func)
    std_double = denoising.compute_std_double(running_map, contamination_arr = contamination_arr, s_cov_func = threshold_func_2fields)

t = time.time() - t
print('Computation completed.')
print(f"Time taken: {t:.2f} seconds")

#TODO change function so that you give it a list of predefined loss functions
image_syn_nu1 = running_map[0]
image_syn_nu2 = running_map[1]

# Convert tuples to NumPy arrays
dust = np.stack([dust_nu1[0], dust_nu2[0]])  # Shape: (2, ...)
data = np.stack([data_nu1[0], data_nu2[0]])  # Shape: (2, ...)
image_denoised = np.stack([image_syn_nu1[0], image_syn_nu2[0]])  # Ensure it's an array

cmb = True
# Create an array of objects to preserve different shapes
results = np.array([dust, data, image_denoised])
np.save(f"nu={nu}_cmb={cmb}", results)