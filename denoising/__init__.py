import os
dirpath = os.path.dirname(__file__)

import numpy as np
# from pathlib import Path
import time
import torch
import sys


from denoising.utils import to_numpy
from denoising.Scattering2d import Scattering2d
from denoising.angle_transforms import FourierAngle, FourierAngleCross
from utils import MBB_factor


def make_denoise_loss_function(
    target, contamination_arr=None, fixed_img=None, std=None, image_init=None, epochNo=None, n_batch=10,
    s_cov_func=None, s_cov_func_2fields=None,
    J=None, L=4, M=None, N=None, l_oversampling=1, frequency_factor=1,
    device='gpu', wavelets='morlet', seed=None,
    if_large_batch=False,
    C11_criteria=None,
    normalization='P00',
    precision='single',
    print_each_step=False,
    pseudo_coef=1,
    remove_edge=False,
    mode='compsep',
    data=None,
    nuisances=None
):
    """
    Factory that constructs and returns the loss function used in `denoise`,
    without running the optimization. This duplicates the internal logic of
    `denoise` so it does not change existing behaviour.
    """

    if not torch.cuda.is_available():
        device = 'cpu'
    np.random.seed(seed)
    if C11_criteria is None:
        C11_criteria = 'j2>=j1'

    # Infer M, N exactly as in `denoise`
    if isinstance(target, tuple):
        _, M, N = target[0].shape
    else:
        _, M, N = target.shape

    # Default init
    if image_init is None:
        image_init = target

    if J is None:
        J = int(np.log2(min(M, N))) - 1

    st_calc = Scattering2d(
        M, N, J, L, device, wavelets,
        l_oversampling=l_oversampling,
        frequency_factor=frequency_factor,
        remove_edge=remove_edge
    )

    def func(map1, ref_map1, map2=None, ref_map2=None):
        coef_list = []

        if map2 is None:
            # Single-field case
            st_calc.add_ref(ref=ref_map1)

            if s_cov_func is None:
                def func_s(x):
                    return st_calc.scattering_cov(
                        x, use_ref=True, if_large_batch=if_large_batch,
                        C11_criteria=C11_criteria,
                        normalization=normalization,
                        pseudo_coef=pseudo_coef,
                        remove_edge=remove_edge
                    )['for_synthesis']
            else:
                def func_s(x):
                    coeffs = st_calc.scattering_cov(
                        x, use_ref=True, if_large_batch=if_large_batch,
                        C11_criteria=C11_criteria,
                        normalization=normalization,
                        pseudo_coef=pseudo_coef,
                        remove_edge=remove_edge
                    )
                    return s_cov_func(coeffs)

            coef_list.append(func_s(map1))

        else:
            # Two-field case
            st_calc.add_ref_ab(ref_a=ref_map1, ref_b=ref_map2)
            if s_cov_func_2fields is None:
                def func_s(x1, x2):
                    coeff_dict = st_calc.scattering_cov_2fields(
                        x1, x2, use_ref=True, if_large_batch=if_large_batch,
                        C11_criteria=C11_criteria,
                        normalization=normalization,
                        remove_edge=remove_edge
                    )
                    return coeff_dict['for_synthesis']
            else:
                def func_s(x1, x2):
                    coeff_dict = st_calc.scattering_cov_2fields(
                        x1, x2, use_ref=True, if_large_batch=if_large_batch,
                        C11_criteria=C11_criteria,
                        normalization=normalization,
                        remove_edge=remove_edge
                    )
                    return s_cov_func_2fields(coeff_dict)

            coef_list.append(func_s(map1, map2))

        return torch.cat(coef_list, axis=-1)


    def loss_func_single(target_single, image_single, std=None, contamination_arr_single=None):
        dtype = torch.double if precision == 'double' else torch.float

        # ---- Convert inputs to torch tensors ----
        if isinstance(target_single, np.ndarray):
            target_single = torch.from_numpy(target_single)
        if isinstance(image_single, np.ndarray):
            image_single = torch.from_numpy(image_single)
        if std is not None and isinstance(std, np.ndarray):
            std = torch.from_numpy(std)

        # Choose device from the global `device` flag
        device_local = torch.device(
            "cuda" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
        )

        target_single = target_single.to(device=device_local, dtype=dtype)
        image_single  = image_single.to(device=device_local, dtype=dtype)
        if std is not None:
            std = std.to(device=device_local, dtype=dtype)
        # -----------------------------------------

        if contamination_arr_single is not None:
            # Sample a batch of contaminations
            n_real = contamination_arr_single.shape[0]
            size = min(n_batch, n_real)
            indices = np.random.choice(n_real, size=size, replace=False)
            contamination_arr_local = contamination_arr_single[indices]

            # Convert contamination to torch on the correct device/dtype
            if isinstance(contamination_arr_local, np.ndarray):
                contamination_arr_local = torch.from_numpy(contamination_arr_local)
            contamination_tensor = contamination_arr_local.to(device=device_local, dtype=dtype)

            # Step 1: Compute reference statistics
            target_stats = func(target_single, target_single).squeeze(0)  # (N_coeffs,)

            # Step 2: Add contamination -> (n_realizations, 1, H, W)
            cont_images = image_single.unsqueeze(0) + contamination_tensor

            # Step 3: Compute noisy statistics in a batched way
            noisy_stats_tensor = func(cont_images[:, 0], target_single)  # (n_realizations, N_coeffs)

            diff = noisy_stats_tensor - target_stats[None, :]

            if std is not None:
                # keep only entries where std > 1e-5
                valid_mask = std > 1e-5

                if valid_mask.any():
                    diff = diff[:, valid_mask]
                    std_local = std[valid_mask]
                    normalized_diff = diff # / std_local[None, :]
                    squared_norms = torch.sum(normalized_diff ** 2, dim=-1) / normalized_diff.size(-1)
                else:
                    squared_norms = torch.sum(diff ** 2, dim=-1) / diff.size(-1)
            else:
                squared_norms = torch.sum(diff ** 2, dim=-1) / diff.size(-1)

        else:
            # No contamination: simple distance between stats of image and target
            target_stats = func(target_single, target_single).squeeze(0).to(dtype=dtype)
            running_stats = func(image_single, target_single).squeeze(0).to(dtype=dtype)
            diff = running_stats - target_stats
            squared_norms = torch.sum(diff ** 2, dim=-1) / diff.size(-1)

        # cleanup
        del contamination_tensor, cont_images, noisy_stats_tensor, diff
        torch.cuda.empty_cache()

        return squared_norms.mean()

    def loss_func_partial(target_p, image_p, fixed_img_p, std=None, contamination_arr_p=None):
        dtype = torch.double if precision == 'double' else torch.float

        # Convert inputs to torch if necessary
        target_local = torch.from_numpy(target_p) if isinstance(target_p, np.ndarray) else target_p
        image_local = torch.from_numpy(image_p) if isinstance(image_p, np.ndarray) else image_p
        fixed_tensor = torch.from_numpy(fixed_img_p) if isinstance(fixed_img_p, np.ndarray) else fixed_img_p

        # Choose device from the global `device` flag
        device_local = torch.device(
            "cuda" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
        )

        target_local = target_local.to(device=device_local, dtype=dtype)
        image_local = image_local.to(device=device_local, dtype=dtype)
        fixed_tensor = fixed_tensor.to(device=device_local, dtype=dtype)

        if contamination_arr_p is not None:
            indices = np.random.choice(contamination_arr_p.shape[0], size=n_batch, replace=False)
            contamination_arr_local = contamination_arr_p[indices]

            if isinstance(contamination_arr_local, np.ndarray):
                contamination_arr_local = torch.from_numpy(contamination_arr_local)

            contamination_tensor = contamination_arr_local.to(device=device_local, dtype=dtype)

            if std is not None:
                if isinstance(std, np.ndarray):
                    std = torch.from_numpy(std)
                std = std_p.to(device=device_local, dtype=dtype)

            target_stats = func(target_local, target_local, fixed_tensor, fixed_tensor).squeeze(0)

            cont_images = image_local.unsqueeze(0) + contamination_tensor
            fixed_batch = fixed_tensor.unsqueeze(0) + torch.zeros_like(contamination_tensor)

            noisy_stats_tensor = func(cont_images[:, 0], target_local, fixed_batch[:, 0], fixed_tensor)

            diff = noisy_stats_tensor - target_stats[None, :]
            valid_mask = std > 1e-6
            diff = diff[:, valid_mask]
            normalized_diff = diff # / std[valid_mask][None, :]
            squared_norms = torch.sum(normalized_diff ** 2, dim=-1) / normalized_diff.size(-1)

            del contamination_tensor, cont_images, fixed_batch, noisy_stats_tensor, diff
            torch.cuda.empty_cache()

            return squared_norms.mean()

        else:
            target_stats = func(target_local, target_local, fixed_tensor, fixed_tensor).squeeze(0).to(dtype=dtype)
            running_stats = func(image_local, target_local, fixed_tensor, fixed_tensor).squeeze(0).to(dtype=dtype)
            diff = running_stats - target_stats
            squared_norms = torch.sum(diff ** 2, dim=-1) / diff.size(-1)
            return squared_norms.mean()

    def loss_func_double(target1, image1, target2, image2, std_double=None, contamination_arr_pair=None):
        dtype = torch.double if precision == 'double' else torch.float

        # Convert inputs to torch if necessary
        if isinstance(target1, np.ndarray):
            target1 = torch.from_numpy(target1)
        if isinstance(image1, np.ndarray):
            image1 = torch.from_numpy(image1)
        if isinstance(target2, np.ndarray):
            target2 = torch.from_numpy(target2)
        if isinstance(image2, np.ndarray):
            image2 = torch.from_numpy(image2)
        if std_double is not None and isinstance(std_double, np.ndarray):
            std_double = torch.from_numpy(std_double)

        # Choose device from the global `device` flag
        device_local = torch.device(
            "cuda" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
        )

        target1 = target1.to(device=device_local, dtype=dtype)
        image1  = image1.to(device=device_local, dtype=dtype)
        target2 = target2.to(device=device_local, dtype=dtype)
        image2  = image2.to(device=device_local, dtype=dtype)
        if std_double is not None:
            std_double = std_double.to(device=device_local, dtype=dtype)

        if contamination_arr_pair is not None and std_double is not None:
            indices = np.random.choice(contamination_arr_pair.shape[0], size=n_batch, replace=False)
            contamination_arr_local = contamination_arr_pair[indices]

            if isinstance(contamination_arr_local, np.ndarray):
                contamination_arr_local = torch.from_numpy(contamination_arr_local)

            contamination_tensor = contamination_arr_local.to(device=device_local, dtype=dtype)

            cont1 = contamination_tensor[:, 0]
            cont2 = contamination_tensor[:, 1]

            target_stats = func(target1, target1, target2, target2).squeeze(0).to(dtype=dtype)

            cont_images1 = image1.unsqueeze(0) + cont1
            cont_images2 = image2.unsqueeze(0) + cont2

            noisy_stats_tensor = func(cont_images1[:, 0], target1, cont_images2[:, 0], target2).to(dtype=dtype)

            diff = noisy_stats_tensor - target_stats[None, :]

            valid_mask = std_double > 1e-6
            diff = diff[:, valid_mask]
            normalized_diff = diff  # / std_double[valid_mask][None, :]
            squared_norms = torch.sum(normalized_diff ** 2, dim=-1) / normalized_diff.size(-1)

        else:
            target_stats = func(target1, target1, target2, target2).squeeze(0).to(dtype=dtype)
            running_stats = func(image1, target1, image2, target2).squeeze(0).to(dtype=dtype)
            diff = running_stats - target_stats
            squared_norms = torch.sum(diff ** 2, dim=-1) / diff.size(-1)

        del contamination_tensor, cont1, cont2, cont_images1, cont_images2, noisy_stats_tensor, diff
        torch.cuda.empty_cache()

        return squared_norms.mean()

    def loss_func_CC(target_cc, image_cc, mean_std=None):
        dtype = torch.double if precision == 'double' else torch.float

        # Choose device from the global `device` flag
        device_local = torch.device(
            "cuda" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
        )

        # Convert inputs to torch if necessary
        if isinstance(target_cc, np.ndarray):
            target_cc = torch.from_numpy(target_cc)
        if isinstance(image_cc, np.ndarray):
            image_cc = torch.from_numpy(image_cc)

        if mean_std is not None:
            target_stats, std_local = mean_std
            target_stats = target_stats.to(device=device_local, dtype=dtype).squeeze(0)
            std_local = std_local.to(device=device_local, dtype=dtype)

            target_cc = target_cc.to(device=device_local, dtype=dtype)
            image_cc  = image_cc.to(device=device_local, dtype=dtype)

            noisy_stats_tensor = func(target_cc - image_cc, target_cc)

            diff = noisy_stats_tensor - target_stats[None, :]

            valid_mask = std_local > 1e-6
            diff = diff[:, valid_mask]
            normalized_diff = diff  # / std_local[valid_mask][None, :]
            squared_norms = torch.sum(normalized_diff ** 2, dim=-1) / normalized_diff.size(-1)

            mean_val = squared_norms.mean()
            if torch.isnan(mean_val):
                return torch.tensor(0.0, device=squared_norms.device, dtype=squared_norms.dtype)
            return mean_val
        else:
            fixed = fixed_img  # assumes `fixed_img` is given to factory
            fixed = torch.from_numpy(fixed) if isinstance(fixed, np.ndarray) else fixed

            fixed     = fixed.to(device=device_local, dtype=dtype)
            target_cc = target_cc.to(device=device_local, dtype=dtype)
            image_cc  = image_cc.to(device=device_local, dtype=dtype)

            ref_stats = func(fixed - target_cc, target_cc).squeeze(0)
            run_stats = func(fixed - image_cc, target_cc).squeeze(0)

            diff = run_stats - ref_stats
            squared_norms = torch.sum(diff ** 2, dim=-1) / diff.size(-1)

            return squared_norms.mean()

    def loss_func_residual(image_r, target_r, data_r, nuisances_r, std_r=None):
        dtype = torch.double if precision == 'double' else torch.float

        image_r = torch.from_numpy(image_r) if isinstance(image_r, np.ndarray) else image_r
        target_r = torch.from_numpy(target_r) if isinstance(target_r, np.ndarray) else target_r
        data_r = torch.from_numpy(data_r) if isinstance(data_r, np.ndarray) else data_r

        device_local = torch.device(
            "cuda" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
        )
        image_r = image_r.to(device=device_local, dtype=dtype)
        target_r = target_r.to(device=device_local, dtype=dtype)
        data_r = data_r.to(device=device_local, dtype=dtype)

        stats_ud = func(image_r, target_r, data_r - image_r, target_r).squeeze(0).to(dtype=dtype)

        if isinstance(nuisances_r, np.ndarray):
            nuisances_r = torch.from_numpy(nuisances_r)
        if isinstance(nuisances_r, list):
            uc_list = []
            for c in nuisances_r:
                c = torch.from_numpy(c) if isinstance(c, np.ndarray) else c
                c = c.to(device=device_local, dtype=dtype)
                uc_list.append(func(image_r, c).squeeze(0).to(dtype=dtype))
            stats_uc_mean = torch.stack(uc_list, dim=0).mean(dim=0)
        else:
            nuisances_r = nuisances_r.to(device=device_local, dtype=dtype)
            uc_list = []
            for k in range(nuisances_r.shape[0]):
                uc_list.append(func(image_r, nuisances_r[k]).squeeze(0).to(dtype=dtype))
            stats_uc_mean = torch.stack(uc_list, dim=0).mean(dim=0)

        diff = stats_ud - stats_uc_mean

        if std_r is not None:
            std_r = torch.from_numpy(std_r) if isinstance(std_r, np.ndarray) else std_r
            std_r = std_r.to(device=device_local, dtype=dtype)
            valid_mask = (std_r > 1e-6)
            diff = diff[valid_mask]

        loss = torch.mean(diff ** 2)
        return loss

    # Now define the outer loss_func that matches the structure inside `denoise`
    if mode == 'compsep':
        def outer_loss_func(*args):
            assert len(args) % 2 == 0, "Expecting equal number of targets and images"
            mid = len(args) // 2
            targets_local, images_local = args[:mid], args[mid:]

            std_single = std['single']
            # std_partial = std['partial']
            # std_double = std['double'][0]
            mean_std = std['noise_mean_std']


            loss_terms = [
                loss_func_single(targets_local[0], images_local[0], std=std_single[0],
                                    contamination_arr_single=contamination_arr[:, 0])
            ]
            # loss_terms = [
            #     loss_func_single(targets_local[0], images_local[0], std=std_single[0],
            #                         contamination_arr_single=contamination_arr[:, 0]),
            #     loss_func_CC(targets_local[0], images_local[0], mean_std[0])
            # ]

            return sum(loss_terms) / len(loss_terms)
    else:
        def outer_loss_func(*args):
            assert len(args) % 2 == 0, "Expecting equal number of targets and images"
            mid = len(args) // 2
            targets_local, images_local = args[:mid], args[mid:]

            loss_terms = [
                loss_func_single(targets_local[0], images_local[0]),
                loss_func_single(targets_local[1], images_local[1]),
                loss_func_double(targets_local[0], images_local[0],
                                 targets_local[1], images_local[1]),
                loss_func_partial(targets_local[0], images_local[0], fixed_img),
                loss_func_partial(targets_local[1], images_local[1], fixed_img),
            ]
            return sum(loss_terms) / len(loss_terms)

    return outer_loss_func

def compute_std(
    target, contamination_arr,
    J=None, L=4, M=None, N=None, l_oversampling=1, frequency_factor=1, s_cov_func = None,
    device='gpu', wavelets='morlet', seed=None,
    if_large_batch=False,
    C11_criteria=None,
    normalization='P00',
    precision='single',
    pseudo_coef=1,
    remove_edge=False
    ):

    '''
the estimator_name can be 's_mean', 's_mean_iso', 's_cov', 's_cov_iso', 'alpha_cov', 
the C11_criteria is the condition on j1 and j2 to compute coefficients, in addition to the condition that j2 >= j1. 
Use * or + to connect more than one condition.
    '''

    if not torch.cuda.is_available(): device='cpu'

    dtype = torch.double if precision == 'double' else torch.float
    if device == 'gpu':
        contamination_arr = torch.tensor(contamination_arr, dtype=dtype).cuda()
    else:
        contamination_arr = torch.tensor(contamination_arr, dtype=dtype)

    np.random.seed(seed)
    if C11_criteria is None:
        C11_criteria = 'j2>=j1'
       
    if isinstance(target, tuple):
        _, M, N = target[0].shape
    else:
        _, M, N = target.shape 
        
    if J is None:
        J = int(np.log2(min(M,N))) - 1
    
    # define calculator and estimator function
    st_calc = Scattering2d(M, N, J, L, device, wavelets, l_oversampling=l_oversampling, frequency_factor=frequency_factor, remove_edge = remove_edge)

    if s_cov_func is None:
        def func_s(x):
            return st_calc.scattering_cov(
                x, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria, 
                normalization=normalization, pseudo_coef=pseudo_coef, remove_edge=remove_edge
            )['for_synthesis']
    else:
        def func_s(x):
            coeff_dict =  st_calc.scattering_cov(
                x, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria, 
                normalization=normalization, pseudo_coef=pseudo_coef, remove_edge=remove_edge
            )
            return s_cov_func(coeff_dict)

    def func(image):
        coef_list = []
        coef_list.append(func_s(image))        
        return torch.cat(coef_list, axis=-1)
                
    def std_func(target_tuple, Mn=10, batch_size=5):
        if device == 'gpu':
            device_name='cuda'
        else:
            device_name=device

        std_list = []

        for i, x in enumerate(target_tuple):
            x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
            x = x.to(device=device_name, dtype=dtype)

            st_calc.add_ref(ref=x)

            cont_i = contamination_arr[:, i].to(device=device_name, dtype=dtype)  # Shape: (Mn, 1, H, W)

            # Compute reference statistics Φ(x)
            coeffs_ref = func(x).squeeze(0)  # Shape: (N_coeffs,)
            coeffs_number = coeffs_ref.size(0)

            # Prepare batches
            batch_number = (Mn + batch_size - 1) // batch_size
            COEFFS = torch.zeros((Mn, coeffs_number), device=device_name, dtype=dtype)

            for b in range(batch_number):
                start_idx = b * batch_size
                end_idx = min((b + 1) * batch_size, Mn)

                cont_batch = cont_i[start_idx:end_idx]  # Shape: (B, 1, H, W)
                cont_images = x.unsqueeze(0) + cont_batch  # Shape: (B, 1, H, W)

                for j in range(cont_images.size(0)):
                    idx = start_idx + j
                    COEFFS[idx] = func(cont_images[j]).squeeze(0)

            std_dev = COEFFS.std(dim=0, unbiased=False)
            std_list.append(std_dev)

        return tuple(std_list)
    
    return std_func(target)


# --- Added function: compute_std_contamination_only ---
def noise_mean_std(
    contamination_arr,
    J=None, L=4, M=None, N=None, l_oversampling=1, frequency_factor=1, s_cov_func=None,
    device='gpu', wavelets='morlet', seed=None,
    if_large_batch=False,
    C11_criteria=None,
    normalization='P00',
    precision='single',
    pseudo_coef=1,
    remove_edge=False
):
    if not torch.cuda.is_available():
        device = 'cpu'

    dtype = torch.double if precision == 'double' else torch.float
    if device == 'gpu':
        contamination_arr = torch.tensor(contamination_arr, dtype=dtype).cuda()
    else:
        contamination_arr = torch.tensor(contamination_arr, dtype=dtype)

    np.random.seed(seed)
    if C11_criteria is None:
        C11_criteria = 'j2>=j1'

    _, M, N = contamination_arr[0, 0].shape

    if J is None:
        J = int(np.log2(min(M, N))) - 1

    st_calc = Scattering2d(M, N, J, L, device, wavelets, l_oversampling=l_oversampling, frequency_factor=frequency_factor, remove_edge = remove_edge)

    if s_cov_func is None:
        def func_s(x):
            return st_calc.scattering_cov(
                x, use_ref=False, if_large_batch=if_large_batch, C11_criteria=C11_criteria,
                normalization=normalization, pseudo_coef=pseudo_coef, remove_edge=remove_edge
            )['for_synthesis']
    else:
        def func_s(x):
            coeff_dict = st_calc.scattering_cov(
                x, use_ref=False, if_large_batch=if_large_batch, C11_criteria=C11_criteria,
                normalization=normalization, pseudo_coef=pseudo_coef, remove_edge=remove_edge
            )
            return s_cov_func(coeff_dict)

    def func(image):
        coef_list = []
        coef_list.append(func_s(image))
        return torch.cat(coef_list, axis=-1)

    def std_func(Mn=10, batch_size=5):
        if device == 'gpu':
            device_name = 'cuda'
        else:
            device_name = device

        std_list = []
        for i in range(contamination_arr.shape[1]):
            cont_i = contamination_arr[:, i].to(device=device_name, dtype=dtype)

            coeffs_number = func(cont_i[0]).squeeze(0).size(0)
            COEFFS = torch.zeros((Mn, coeffs_number), device=device_name, dtype=dtype)

            batch_number = (Mn + batch_size - 1) // batch_size
            for b in range(batch_number):
                start_idx = b * batch_size
                end_idx = min((b + 1) * batch_size, Mn)

                cont_batch = cont_i[start_idx:end_idx]
                for j in range(cont_batch.size(0)):
                    idx = start_idx + j
                    COEFFS[idx] = func(cont_batch[j]).squeeze(0)

            std_dev = COEFFS.std(dim=0, unbiased=False)
            mean_val = COEFFS.mean(dim=0)
            std_list.append((mean_val, std_dev))

        return tuple(std_list)

    return std_func()


def compute_std_partial(
    target, contamination_arr, fixed_img,
    J=None, L=4, M=None, N=None, l_oversampling=1, frequency_factor=1, s_cov_func = None,
    device='gpu', wavelets='morlet', seed=None,
    if_large_batch=False,
    C11_criteria=None,
    normalization='P00',
    precision='single',
    remove_edge=False
    ):

    '''
the estimator_name can be 's_mean', 's_mean_iso', 's_cov', 's_cov_iso', 'alpha_cov', 
the C11_criteria is the condition on j1 and j2 to compute coefficients, in addition to the condition that j2 >= j1. 
Use * or + to connect more than one condition.
    '''

    if not torch.cuda.is_available(): device='cpu'

    dtype = torch.double if precision == 'double' else torch.float
    if device == 'gpu':
        contamination_arr = torch.tensor(contamination_arr, dtype=dtype).cuda()
    else:
        contamination_arr = torch.tensor(contamination_arr, dtype=dtype)

    # Ensure fixed_img is a torch tensor on the same device/dtype
    if isinstance(fixed_img, np.ndarray):
        fixed_img = torch.from_numpy(fixed_img)
    if device == 'gpu':
        device_name_fixed = 'cuda'
    else:
        device_name_fixed = device
    fixed_img = fixed_img.to(device=device_name_fixed, dtype=dtype)

    np.random.seed(seed)
    if C11_criteria is None:
        C11_criteria = 'j2>=j1'
       
    if isinstance(target, tuple):
        _, M, N = target[0].shape
    else:
        _, M, N = target.shape 
        
    if J is None:
        J = int(np.log2(min(M,N))) - 1
        
    if s_cov_func is None: 
        def func_s(x):
                st_calc.add_ref_ab(ref_a=x, ref_b=fixed_img)
                coeff_dict = st_calc.scattering_cov_2fields(
                    x, fixed_img, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria,
                    normalization=normalization, remove_edge=remove_edge
                )
                return coeff_dict['for_synthesis']
    else:
        def func_s(x):
                st_calc.add_ref_ab(ref_a=x, ref_b=fixed_img)
                coeff_dict = st_calc.scattering_cov_2fields(
                    x, fixed_img, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria,
                    normalization=normalization, remove_edge=remove_edge
                )
                return s_cov_func(coeff_dict)

    st_calc = Scattering2d(M, N, J, L, device, wavelets, l_oversampling=l_oversampling, frequency_factor=frequency_factor, remove_edge=remove_edge)
    def func(image):
        coef_list = []
        coef_list.append(func_s(image))
        return torch.cat(coef_list, axis=-1)
                    
    def std_func(target_tuple, Mn=10, batch_size=5):
        device_name = device_name_fixed

        std_list = []

        for i, x in enumerate(target_tuple):
            x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
            x = x.to(device=device_name, dtype=dtype)

            cont_i = contamination_arr[:, i].to(device=device_name, dtype=dtype)  # Shape: (Mn, 1, H, W)

            # Compute reference statistics Φ(x)
            coeffs_ref = func(x).squeeze(0)  # Shape: (N_coeffs,)
            coeffs_number = coeffs_ref.size(0)

            # Prepare batches
            batch_number = (Mn + batch_size - 1) // batch_size
            COEFFS = torch.zeros((Mn, coeffs_number), device=device_name, dtype=dtype)

            for b in range(batch_number):
                start_idx = b * batch_size
                end_idx = min((b + 1) * batch_size, Mn)

                cont_batch = cont_i[start_idx:end_idx]  # Shape: (B, 1, H, W)
                cont_images = x.unsqueeze(0) + cont_batch  # Shape: (B, 1, H, W)

                for j in range(cont_images.size(0)):
                    idx = start_idx + j
                    COEFFS[idx] = func(cont_images[j]).squeeze(0)

            std_dev = COEFFS.std(dim=0, unbiased=False)
            std_list.append(std_dev)

        return tuple(std_list)
    
    return std_func(target)

def compute_std_double(
    image, contamination_arr, image_ref=None, s_cov_func = None, 
    J=None, L=4, M=None, N=None, l_oversampling=1, frequency_factor=1,
    device='gpu', wavelets='morlet', seed=None,
    if_large_batch=False,
    C11_criteria=None,
    normalization='P00',
    precision='single',
    remove_edge=False,
):
    '''
the estimator_name can be 's_mean', 's_mean_iso', 's_cov', 's_cov_iso', 'alpha_cov', 
the C11_criteria is the condition on j1 and j2 to compute coefficients, in addition to the condition that j2 >= j1. 
Use * or + to connect more than one condition.
    '''
    if not torch.cuda.is_available(): device='cpu'
    np.random.seed(seed)
    C11_criteria = 'j2>=j1'
    _, M, N = image[0].shape
    
    J = int(np.log2(min(M,N))) - 1 

    image_ref = image if image_ref is None else image_ref
    # Insert assertion after image_ref assignment
    assert contamination_arr.shape[1] == len(image), \
        f"Number of image channels ({len(image)}) does not match contamination channels ({contamination_arr.shape[1]})"

    if s_cov_func is None: 
        def func_s(x1, x2):
                coeff_dict = st_calc.scattering_cov_2fields(
                    x1, x2, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria,
                    normalization=normalization, remove_edge=remove_edge
                )
                return coeff_dict['for_synthesis']
    else:
        def func_s(x1, x2):
                coeff_dict = st_calc.scattering_cov_2fields(
                    x1, x2, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria,
                    normalization=normalization, remove_edge=remove_edge
                )
                return s_cov_func(coeff_dict)

    st_calc = Scattering2d(M, N, J, L, device, wavelets, l_oversampling=l_oversampling, frequency_factor=frequency_factor, remove_edge = remove_edge)
    def func(map1, ref_map1, map2=None, ref_map2=None):
        coef_list = []
        # Two-field case
        st_calc.add_ref_ab(ref_a=ref_map1, ref_b=ref_map2)
        coef_list.append(func_s(map1, map2))
        return torch.cat(coef_list, axis=-1)

    def std_func_dual(x1, ref1, x2, ref2, contamination_arr_pair, Mn=10, batch_size=5):
        # Convert inputs to torch tensors if needed
        x1 = torch.from_numpy(x1) if isinstance(x1, np.ndarray) else x1
        x2 = torch.from_numpy(x2) if isinstance(x2, np.ndarray) else x2
        ref1 = torch.from_numpy(ref1) if isinstance(ref1, np.ndarray) else ref1
        ref2 = torch.from_numpy(ref2) if isinstance(ref2, np.ndarray) else ref2
        # Set dtype and device
        dtype = torch.double if precision == 'double' else torch.float
        device_torch = torch.device('cuda' if device == 'gpu' else 'cpu')
        x1 = x1.to(device=device_torch, dtype=dtype)
        x2 = x2.to(device=device_torch, dtype=dtype)
        ref1 = ref1.to(device=device_torch, dtype=dtype)
        ref2 = ref2.to(device=device_torch, dtype=dtype)
        contamination_tensor = torch.from_numpy(contamination_arr_pair).to(device=device_torch, dtype=dtype)
        # Extract contamination for both inputs
        cont1 = contamination_tensor[:, 0]  # Shape: (Mn, 1, H, W)
        cont2 = contamination_tensor[:, 1]  # Shape: (Mn, 1, H, W)
        # Split into batches
        batch_number = (Mn + batch_size - 1) // batch_size
        COEFFS = []
        for i in range(batch_number):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, Mn)
            x1_noisy_batch = x1.unsqueeze(0) + cont1[start_idx:end_idx]
            x2_noisy_batch = x2.unsqueeze(0) + cont2[start_idx:end_idx]
            for j in range(end_idx - start_idx):
                stats = func(x1_noisy_batch[j], ref1, x2_noisy_batch[j], ref2).squeeze(0)
                COEFFS.append(stats)
        COEFFS = torch.stack(COEFFS, dim=0)  # Shape: (Mn, N_coeffs)
        std_dev = COEFFS.std(dim=0, unbiased=False)
        return std_dev

    # Generate std_func_dual for all unique pairs
    from itertools import combinations
    stds = []
    for i, j in combinations(range(len(image)), 2):
        std_ij = std_func_dual(image[i], image_ref[i], image[j], image_ref[j],
                            contamination_arr[:, [i, j]])
        stds.append(std_ij)
    return tuple(stds)

def denoise(
    target, contamination_arr=None, fixed_img=None, std=None, image_init=None, epochNo=None, n_batch=10,
    s_cov_func=None, s_cov_func_2fields=None,
    J=None, L=4, M=None, N=None, l_oversampling=1, frequency_factor=1,
    optim_algorithm='LBFGS', steps=300, learning_rate=0.2,
    device='gpu', wavelets='morlet', seed=None,
    if_large_batch=False,
    C11_criteria=None,
    normalization='P00',
    precision='single',
    print_each_step=False,
    pseudo_coef=1,
    remove_edge=False,
    mode='compsep',
    data=None,
    nuisances=None
):
    """
    Denoising wrapper around `denoise_general` that now uses *exactly* the same
    loss definitions as `make_denoise_loss_function`, so that for the same
    inputs (target, contamination_arr, std, etc.) the loss being optimized is
    identical to the one built by the factory.
    """

    if not torch.cuda.is_available():
        device = 'cpu'
    np.random.seed(seed)
    if C11_criteria is None:
        C11_criteria = 'j2>=j1'

    # Infer M, N as in make_denoise_loss_function
    if isinstance(target, tuple):
        _, M, N = target[0].shape
    else:
        _, M, N = target.shape

    # Initial point of synthesis
    if image_init is None:
        image_init = target

    if J is None:
        J = int(np.log2(min(M, N))) - 1

    # Scattering calculator
    st_calc = Scattering2d(
        M, N, J, L, device, wavelets,
        l_oversampling=l_oversampling,
        frequency_factor=frequency_factor,
        remove_edge=remove_edge
    )

    # --- same func as in make_denoise_loss_function ---
    def func(map1, ref_map1, map2=None, ref_map2=None):
        coef_list = []

        if map2 is None:
            # Single-field case
            st_calc.add_ref(ref=ref_map1)

            if s_cov_func is None:
                def func_s(x):
                    return st_calc.scattering_cov(
                        x, use_ref=True, if_large_batch=if_large_batch,
                        C11_criteria=C11_criteria,
                        normalization=normalization,
                        pseudo_coef=pseudo_coef,
                        remove_edge=remove_edge
                    )['for_synthesis']
            else:
                def func_s(x):
                    coeffs = st_calc.scattering_cov(
                        x, use_ref=True, if_large_batch=if_large_batch,
                        C11_criteria=C11_criteria,
                        normalization=normalization,
                        pseudo_coef=pseudo_coef,
                        remove_edge=remove_edge
                    )
                    return s_cov_func(coeffs)

            coef_list.append(func_s(map1))

        else:
            # Two-field case
            st_calc.add_ref_ab(ref_a=ref_map1, ref_b=ref_map2)
            if s_cov_func_2fields is None:
                def func_s(x1, x2):
                    coeff_dict = st_calc.scattering_cov_2fields(
                        x1, x2, use_ref=True, if_large_batch=if_large_batch,
                        C11_criteria=C11_criteria,
                        normalization=normalization,
                        remove_edge=remove_edge
                    )
                    return coeff_dict['for_synthesis']
            else:
                def func_s(x1, x2):
                    coeff_dict = st_calc.scattering_cov_2fields(
                        x1, x2, use_ref=True, if_large_batch=if_large_batch,
                        C11_criteria=C11_criteria,
                        normalization=normalization,
                        remove_edge=remove_edge
                    )
                    return s_cov_func_2fields(coeff_dict)

            coef_list.append(func_s(map1, map2))

        return torch.cat(coef_list, axis=-1)

    # --- loss functions copied from make_denoise_loss_function ---

    def loss_func_single(target_single, image_single, std=None, contamination_arr=None):
        dtype = torch.double if precision == 'double' else torch.float

        # Convert inputs to torch tensors
        if isinstance(target_single, np.ndarray):
            target_single = torch.from_numpy(target_single)
        if isinstance(image_single, np.ndarray):
            image_single = torch.from_numpy(image_single)
        if std is not None and isinstance(std, np.ndarray):
            std = torch.from_numpy(std)

        # Choose device from global flag
        device_local = torch.device(
            "cuda" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
        )

        target_single = target_single.to(device=device_local, dtype=dtype)
        image_single  = image_single.to(device=device_local, dtype=dtype)
        if std is not None:
            std = std.to(device=device_local, dtype=dtype)

        if contamination_arr is not None:
            # Sample a batch of contaminations
            n_real = contamination_arr.shape[0]
            size = min(n_batch, n_real)
            indices = np.random.choice(n_real, size=size, replace=False)
            contamination_arr_local = contamination_arr[indices]

            # Convert contamination to torch
            if isinstance(contamination_arr_local, np.ndarray):
                contamination_arr_local = torch.from_numpy(contamination_arr_local)
            contamination_tensor = contamination_arr_local.to(device=device_local, dtype=dtype)

            # Reference stats
            target_stats = func(target_single, target_single).squeeze(0)

            # Add contamination
            cont_images = image_single.unsqueeze(0) + contamination_tensor

            # Noisy stats
            noisy_stats_tensor = func(cont_images[:, 0], target_single)

            diff = noisy_stats_tensor - target_stats[None, :]

            if std is not None:
                valid_mask = std > 1e-5
                if valid_mask.any():
                    diff = diff[:, valid_mask]
                    std_local = std[valid_mask]
                    normalized_diff = diff #/ std_local[None, :]
                    squared_norms = torch.sum(normalized_diff ** 2, dim=-1) / normalized_diff.size(-1)
                else:
                    squared_norms = torch.sum(diff ** 2, dim=-1) / diff.size(-1)
            else:
                squared_norms = torch.sum(diff ** 2, dim=-1) / diff.size(-1)

        else:
            # No contamination: simple distance
            target_stats = func(target_single, target_single).squeeze(0).to(dtype=dtype)
            running_stats = func(image_single, target_single).squeeze(0).to(dtype=dtype)
            diff = running_stats - target_stats
            squared_norms = torch.sum(diff ** 2, dim=-1) / diff.size(-1)

        return squared_norms.mean()

    def loss_func_partial(target_p, image_p, fixed_img_p, std=None, contamination_arr=None):
        dtype = torch.double if precision == 'double' else torch.float

        target_local = torch.from_numpy(target_p) if isinstance(target_p, np.ndarray) else target_p
        image_local = torch.from_numpy(image_p) if isinstance(image_p, np.ndarray) else image_p
        fixed_tensor = torch.from_numpy(fixed_img_p) if isinstance(fixed_img_p, np.ndarray) else fixed_img_p

        device_local = torch.device(
            "cuda" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
        )

        target_local = target_local.to(device=device_local, dtype=dtype)
        image_local = image_local.to(device=device_local, dtype=dtype)
        fixed_tensor = fixed_tensor.to(device=device_local, dtype=dtype)

        if contamination_arr is not None:
            indices = np.random.choice(contamination_arr.shape[0], size=n_batch, replace=False)
            contamination_arr_local = contamination_arr[indices]

            if isinstance(contamination_arr_local, np.ndarray):
                contamination_arr_local = torch.from_numpy(contamination_arr_local)

            contamination_tensor = contamination_arr_local.to(device=device_local, dtype=dtype)

            if std is not None:
                if isinstance(std, np.ndarray):
                    std = torch.from_numpy(std)
                std = std.to(device=device_local, dtype=dtype)

            target_stats = func(target_local, target_local, fixed_tensor, fixed_tensor).squeeze(0)

            cont_images = image_local.unsqueeze(0) + contamination_tensor
            fixed_batch = fixed_tensor.unsqueeze(0) + torch.zeros_like(contamination_tensor)

            noisy_stats_tensor = func(cont_images[:, 0], target_local, fixed_batch[:, 0], fixed_tensor)

            diff = noisy_stats_tensor - target_stats[None, :]
            valid_mask = std > 1e-6
            diff = diff[:, valid_mask]
            normalized_diff = diff  #/ std[valid_mask][None, :]
            squared_norms = torch.sum(normalized_diff ** 2, dim=-1) / normalized_diff.size(-1)

            return squared_norms.mean()

        else:
            target_stats = func(target_local, target_local, fixed_tensor, fixed_tensor).squeeze(0).to(dtype=dtype)
            running_stats = func(image_local, target_local, fixed_tensor, fixed_tensor).squeeze(0).to(dtype=dtype)
            diff = running_stats - target_stats
            squared_norms = torch.sum(diff ** 2, dim=-1) / diff.size(-1)
            return squared_norms.mean()

    def loss_func_double(target1, image1, target2, image2, std_double=None, contamination_arr=None):
        dtype = torch.double if precision == 'double' else torch.float

        if isinstance(target1, np.ndarray):
            target1 = torch.from_numpy(target1)
        if isinstance(image1, np.ndarray):
            image1 = torch.from_numpy(image1)
        if isinstance(target2, np.ndarray):
            target2 = torch.from_numpy(target2)
        if isinstance(image2, np.ndarray):
            image2 = torch.from_numpy(image2)
        if std_double is not None and isinstance(std_double, np.ndarray):
            std_double = torch.from_numpy(std_double)

        device_local = torch.device(
            "cuda" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
        )

        target1 = target1.to(device=device_local, dtype=dtype)
        image1  = image1.to(device=device_local, dtype=dtype)
        target2 = target2.to(device=device_local, dtype=dtype)
        image2  = image2.to(device=device_local, dtype=dtype)
        if std_double is not None:
            std_double = std_double.to(device=device_local, dtype=dtype)

        if contamination_arr is not None and std_double is not None:
            indices = np.random.choice(contamination_arr.shape[0], size=n_batch, replace=False)
            contamination_arr_local = contamination_arr[indices]

            if isinstance(contamination_arr_local, np.ndarray):
                contamination_arr_local = torch.from_numpy(contamination_arr_local)

            contamination_tensor = contamination_arr_local.to(device=device_local, dtype=dtype)

            cont1 = contamination_tensor[:, 0]
            cont2 = contamination_tensor[:, 1]

            target_stats = func(target1, target1, target2, target2).squeeze(0).to(dtype=dtype)

            cont_images1 = image1.unsqueeze(0) + cont1
            cont_images2 = image2.unsqueeze(0) + cont2

            noisy_stats_tensor = func(cont_images1[:, 0], target1, cont_images2[:, 0], target2).to(dtype=dtype)

            diff = noisy_stats_tensor - target_stats[None, :]

            valid_mask = std_double > 1e-6
            diff = diff[:, valid_mask]
            normalized_diff = diff #/ std_double[valid_mask][None, :]
            squared_norms = torch.sum(normalized_diff ** 2, dim=-1) / normalized_diff.size(-1)

        else:
            target_stats = func(target1, target1, target2, target2).squeeze(0).to(dtype=dtype)
            running_stats = func(image1, target1, image2, target2).squeeze(0).to(dtype=dtype)
            diff = running_stats - target_stats
            squared_norms = torch.sum(diff ** 2, dim=-1) / diff.size(-1)

        return squared_norms.mean()

    def loss_func_CC(target_cc, image_cc, mean_std=None):
        dtype = torch.double if precision == 'double' else torch.float

        device_local = torch.device(
            "cuda" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
        )

        if isinstance(target_cc, np.ndarray):
            target_cc = torch.from_numpy(target_cc)
        if isinstance(image_cc, np.ndarray):
            image_cc = torch.from_numpy(image_cc)

        if mean_std is not None:
            target_stats, std_local = mean_std
            target_stats = target_stats.to(device=device_local, dtype=dtype).squeeze(0)
            std_local = std_local.to(device=device_local, dtype=dtype)

            target_cc = target_cc.to(device=device_local, dtype=dtype)
            image_cc  = image_cc.to(device=device_local, dtype=dtype)

            noisy_stats_tensor = func(target_cc - image_cc, target_cc)

            diff = noisy_stats_tensor - target_stats[None, :]

            valid_mask = std_local > 1e-6
            diff = diff[:, valid_mask]
            normalized_diff = diff #/ std_local[valid_mask][None, :]
            squared_norms = torch.sum(normalized_diff ** 2, dim=-1) / normalized_diff.size(-1)

            mean_val = squared_norms.mean()
            if torch.isnan(mean_val):
                return torch.tensor(0.0, device=squared_norms.device, dtype=squared_norms.dtype)
            return mean_val
        else:
            fixed = fixed_img
            fixed = torch.from_numpy(fixed) if isinstance(fixed, np.ndarray) else fixed

            fixed     = fixed.to(device=device_local, dtype=dtype)
            target_cc = target_cc.to(device=device_local, dtype=dtype)
            image_cc  = image_cc.to(device=device_local, dtype=dtype)

            ref_stats = func(fixed - target_cc, target_cc).squeeze(0)
            run_stats = func(fixed - image_cc, target_cc).squeeze(0)

            diff = run_stats - ref_stats
            squared_norms = torch.sum(diff ** 2, dim=-1) / diff.size(-1)

            return squared_norms.mean()

    def loss_func_residual(image_r, target_r, data_r, nuisances_r, std_r=None):
        dtype = torch.double if precision == 'double' else torch.float

        image_r = torch.from_numpy(image_r) if isinstance(image_r, np.ndarray) else image_r
        target_r = torch.from_numpy(target_r) if isinstance(target_r, np.ndarray) else target_r
        data_r = torch.from_numpy(data_r) if isinstance(data_r, np.ndarray) else data_r

        device_local = torch.device(
            "cuda" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
        )
        image_r = image_r.to(device=device_local, dtype=dtype)
        target_r = target_r.to(device=device_local, dtype=dtype)
        data_r = data_r.to(device=device_local, dtype=dtype)

        stats_ud = func(image_r, target_r, data_r - image_r, target_r).squeeze(0).to(dtype=dtype)

        if isinstance(nuisances_r, np.ndarray):
            nuisances_r = torch.from_numpy(nuisances_r)
        if isinstance(nuisances_r, list):
            uc_list = []
            for c in nuisances_r:
                c = torch.from_numpy(c) if isinstance(c, np.ndarray) else c
                c = c.to(device=device_local, dtype=dtype)
                uc_list.append(func(image_r, c).squeeze(0).to(dtype=dtype))
            stats_uc_mean = torch.stack(uc_list, dim=0).mean(dim=0)
        else:
            nuisances_r = nuisances_r.to(device=device_local, dtype=dtype)
            uc_list = []
            for k in range(nuisances_r.shape[0]):
                uc_list.append(func(image_r, nuisances_r[k]).squeeze(0).to(dtype=dtype))
            stats_uc_mean = torch.stack(uc_list, dim=0).mean(dim=0)

        diff = stats_ud - stats_uc_mean

        if std_r is not None:
            std_r = torch.from_numpy(std_r) if isinstance(std_r, np.ndarray) else std_r
            std_r = std_r.to(device=device_local, dtype=dtype)
            valid_mask = (std_r > 1e-6)
            diff = diff[valid_mask]

        loss = torch.mean(diff ** 2)
        return loss

    # --- outer loss_func with epoch-dependent logic (single vs CC term) ---

    if mode == 'compsep':            
        def loss_func(*args):
            assert len(args) % 2 == 0, "Expecting equal number of targets and images"
            mid = len(args) // 2
            targets, images = args[:mid], args[mid:]

            std_single = std['single']
            std_partial = std['partial']
            std_double = std['double'][0]
            mean_std = std['noise_mean_std']

            if epochNo is None or epochNo % 2 == 0:
                loss_terms = [
                    loss_func_single(targets[0], images[0], std = std_single[0], contamination_arr = contamination_arr[:, 0]),
                    loss_func_single(targets[1], images[1], std = std_single[1], contamination_arr = contamination_arr[:, 1]),
                    loss_func_partial(targets[0], images[0], fixed_img, std_partial[0], contamination_arr[:, 0]),
                    loss_func_partial(targets[0], images[0], fixed_img, std_partial[1], contamination_arr[:, 1]),
                    loss_func_double(targets[0], images[0], targets[1], images[1], std_double, contamination_arr)
                ]
            else:
                loss_terms = [
                    loss_func_single(targets[0], images[0], std = std_single[0], contamination_arr = contamination_arr[:, 0]),
                    loss_func_single(targets[1], images[1], std = std_single[1], contamination_arr = contamination_arr[:, 1]),
                    loss_func_partial(targets[0], images[0], fixed_img, std_partial[0], contamination_arr[:, 0]),
                    loss_func_partial(targets[0], images[0], fixed_img, std_partial[1], contamination_arr[:, 1]),
                    loss_func_double(targets[0], images[0], targets[1], images[1], std_double, contamination_arr),
                    loss_func_CC(targets[0], images[0], mean_std[0]),
                    loss_func_CC(targets[1], images[1], mean_std[1])
                ]
            
            return sum(loss_terms) / len(loss_terms)
    else:
        def loss_func(*args):
            assert len(args) % 2 == 0, "Expecting equal number of targets and images"
            mid = len(args) // 2
            targets_local, images_local = args[:mid], args[mid:]

            loss_terms = [
                loss_func_single(targets_local[0], images_local[0]),
                loss_func_single(targets_local[1], images_local[1]),
                loss_func_double(targets_local[0], images_local[0],
                                 targets_local[1], images_local[1]),
                loss_func_partial(targets_local[0], images_local[0], fixed_img),
                loss_func_partial(targets_local[1], images_local[1], fixed_img),
            ]
            return sum(loss_terms) / len(loss_terms)

    # --- run the optimizer ---
    image_syn = denoise_general(
        target, image_init, func, loss_func,
        optim_algorithm=optim_algorithm, steps=steps, learning_rate=learning_rate,
        device=device, precision=precision, print_each_step=print_each_step
    )

    return image_syn

def denoise_general(
    target, image_init, estimator_function, loss_function, 
    optim_algorithm='LBFGS', steps=100, learning_rate=0.5,
    device='gpu', precision='single', print_each_step=False
):    
    # Formatting targets and images (to tensor, to CUDA if necessary)
    def to_tensor(var):
        if isinstance(var, np.ndarray):
            var = torch.from_numpy(var)
        if precision == 'double':
            var = var.type(torch.DoubleTensor)
        else:
            var = var.type(torch.FloatTensor)
        if device == 'gpu':
            var = var.cuda()
        return var
    
    targets = [to_tensor(t) for t in target]
    image_inits = [to_tensor(img) for img in image_init]

    # # calculate statistics for target images
    # estimator_single = estimator_function(*targets)
    # estimator_double = estimator_function(*targets, *targets)
    
    # print('# of estimators: ', estimator_single.shape[-1] + estimator_double.shape[-1])
    
    # Define optimizable image model
    class OptimizableImage(torch.nn.Module):
        def __init__(self, input_inits):
            super().__init__()
            self.params = torch.nn.ParameterList([
                torch.nn.Parameter(img) for img in input_inits
            ])

        def get_images(self):
            return list(self.params)

    # Ensure inputs are on the correct device and with the correct precision
    for i in range(len(image_inits)):
        image_inits[i] = image_inits[i].double() if precision == 'double' else image_inits[i].float()
        if device == 'gpu':
            image_inits[i] = image_inits[i].cuda()

    # Initialize the model
    image_model = OptimizableImage(image_inits)
        
    # Define optimizer for both image parameters
    optimizer = torch.optim.LBFGS(
        image_model.parameters(), lr=learning_rate, 
        max_iter=steps, max_eval=None, 
        tolerance_grad=1e-19, tolerance_change=1e-19, 
        history_size=min(steps // 2, 150), line_search_fn=None
    )

    # Define closure for LBFGS optimizer
    def closure():
        optimizer.zero_grad()

        # Retrieve the synthesized images
        synthesized_images = image_model.get_images()

        # Compute the loss using the loss function with all targets and synthesized images
        loss = loss_function(*targets, *synthesized_images)

        # Check for NaN loss
        if torch.isnan(loss):
            raise RuntimeError("Loss is NaN! Terminating process...")

        # Print progress if required 
        if print_each_step:
            print(f'Current Loss: {loss.item():.2e}')

        # Backpropagate the loss
        loss.backward()
        return loss
    
    # Perform optimization
    t_start = time.time()
    if optim_algorithm == 'LBFGS':
        optimizer.step(closure)
    else:
        for i in range(steps):
            optimizer.step(closure)
    t_end = time.time()
    print('Time used: ', t_end - t_start, 's')

    # Return the optimized images as numpy arrays
    return tuple(img.cpu().detach().numpy() for img in image_model.get_images())


def scale_annotation_a_b(idx_info):
    """
    Convert idx_info j1, j1p, j2, l1, l1p, l2
    into idx_info j1, a, b, l1, l1p, l2.

    :idx_info: K x 6 array
    """
    cov_type, j1, j1p, j2, l1, l1p, l2 = idx_info.T
    admissible_types = {
        0: 'mean',
        1: 'P00',
        2: 'S1',
        3: 'C01re',
        4: 'C01im',
        5: 'C11re',
        6: 'C11im'
    }
    cov_type = np.array([admissible_types[c_type] for c_type in cov_type])

    # create idx_info j1, j1p, a, b, l1, l1p, l2
    where_c01_c11 = np.isin(cov_type, ['C01re', 'C01im', 'C11re', 'C11im'])

    j1_new = j1.copy()
    j1p_new = j1p.copy()

    j1_new[where_c01_c11] = j1p[where_c01_c11]
    j1p_new[where_c01_c11] = j1[where_c01_c11]

    a = (j1_new - j1p_new) * (j1p_new >= 0) - (j1p_new == -1)
    b = (j1_new - j2) * (j2 >= 0) + (j2 == -1)

    idx_info_a_b = np.array([cov_type, j1_new, a, b, l1, l1p, l2], dtype=object).T

    # idx_info_a_b = np.stack([cov_type, j1_new, a, b, l1, l1p, l2]).T

    return idx_info_a_b

# def scale_annotation_a_b_2fields(idx_info):
#     """
#     Convert extended idx_info for 2-field scattering coefficients into a-b scale notation.

#     Input:
#         idx_info: K x 7 array
#             Columns: [cov_type_code, j1, j1p, j2, l1, l1p, l2]

#     Output:
#         idx_info_a_b: K x 7 array
#             Columns: [cov_type_str, j1, a, b, l1, l1p, l2]
#     """
#     cov_type_codes, j1, j1p, j2, l1, l1p, l2 = idx_info.T.astype(object)

#     # Define code mapping based on your setup
#     cov_type_map = {
#         0: 'mean',         1: 'P00',        2: 'S1',
#         7: 'Corr00re',     8: 'Corr00im',
#         9: 'C01a_re',     10: 'C01a_im',
#         11: 'C01b_re',    12: 'C01b_im',
#         13: 'C01ab_re',   14: 'C01ab_im',
#         15: 'C01ba_re',   16: 'C01ba_im',
#         17: 'Corr11aa_re', 18: 'Corr11aa_im',
#         19: 'Corr11bb_re', 20: 'Corr11bb_im',
#         21: 'Corr11ab_re', 22: 'Corr11ab_im',
#     }

#     # Map int code to string label
#     cov_type_strs = np.array([cov_type_map.get(code, f"UNK_{code}") for code in cov_type_codes])

#     # By default, j1 is taken as is. For "C01..." and "Corr11..." types, swap j1 <-> j1p if needed
#     is_c01_c11 = np.array([
#     s.startswith('C01') or s.startswith('Corr11') for s in cov_type_strs
#     ])

#     j1_new = j1.copy()
#     j1p_new = j1p.copy()

#     j1_new[is_c01_c11] = j1p[is_c01_c11]
#     j1p_new[is_c01_c11] = j1[is_c01_c11]

#     # Compute a and b
#     a = (j1_new - j1p_new) * (j1p_new >= 0) - (j1p_new == -1)
#     b = (j1_new - j2) * (j2 >= 0) + (j2 == -1)

#     idx_info_a_b = np.array([cov_type_strs, j1_new, a, b, l1, l1p, l2], dtype=object).T

#     return idx_info_a_b

def scale_annotation_a_b_2fields(idx_info):
    """
    Convert extended idx_info for 2-field scattering coefficients into a-b scale notation.

    Input:
        idx_info: K x 7 array
            Columns: [cov_type_code, j1, j1p, j2, l1, l1p, l2]

    Output:
        idx_info_a_b: K x 7 array
            Columns: [cov_type_str, j1, a, b, l1, l1p, l2]
            For non-angular coefficients, l1, l1p, l2 are set to 0 to avoid FFT problems.
    """
    cov_type_codes, j1, j1p, j2, l1, l1p, l2 = idx_info.T.astype(object)

    cov_type_map = {
        0: 'mean',         1: 'P00',        2: 'S1',
        7: 'Corr00re',     8: 'Corr00im',
        9: 'C01a_re',     10: 'C01a_im',
        11: 'C01b_re',    12: 'C01b_im',
        13: 'C01ab_re',   14: 'C01ab_im',
        15: 'C01ba_re',   16: 'C01ba_im',
        17: 'Corr11aa_re', 18: 'Corr11aa_im',
        19: 'Corr11bb_re', 20: 'Corr11bb_im',
        21: 'Corr11ab_re', 22: 'Corr11ab_im',
    }

    cov_type_strs = np.array([cov_type_map.get(code, f"UNK_{code}") for code in cov_type_codes])

    # Flag angular types
    is_c01_or_c11 = np.array([
        s.startswith('C01') or s.startswith('Corr11') for s in cov_type_strs
    ])

    # Swap j1 <-> j1p for these
    j1_new = j1.copy()
    j1p_new = j1p.copy()
    j1_new[is_c01_or_c11] = j1p[is_c01_or_c11]
    j1p_new[is_c01_or_c11] = j1[is_c01_or_c11]

    # Compute a, b
    a = (j1_new - j1p_new) * (j1p_new >= 0) - (j1p_new == -1)
    b = (j1_new - j2) * (j2 >= 0) + (j2 == -1)

    # Only keep angular indices if needed
    l1_out = np.zeros_like(l1)
    l1p_out = np.zeros_like(l1p)
    l2_out = np.zeros_like(l2)

    l1_out[is_c01_or_c11] = l1[is_c01_or_c11]
    l1p_out[is_c01_or_c11] = l1p[is_c01_or_c11]
    l2_out[is_c01_or_c11] = l2[is_c01_or_c11]

    # Construct final output (all columns present, safe values for non-angular types)
    idx_info_a_b = np.array([cov_type_strs, j1_new, a, b, l1_out, l1p_out, l2_out], dtype=object).T

    return idx_info_a_b

def filter_radial(img, func, backend='np'):
        M, N = img.shape[-2:]
        X = np.arange(M)[:,None]
        Y = np.arange(N)[None,:]
        R = ((X-M//2)**2+(Y-N//2)**2)**0.5
        if len(img.shape)==2:
            filter = func(R)
        else:
            filter = func(R)[None,:,:]
        if backend=='np':
            img_f = np.fft.fft2(img)
            img_filtered = np.fft.ifft2(
                np.fft.ifftshift(filter, axes=(-2,-1)) * img_f
            ).real
        if backend=='torch':
            # Ensure filter is a torch tensor on the same device/dtype as img
            if not torch.is_tensor(filter):
                filter_torch = torch.from_numpy(filter).to(device=img.device, dtype=img.dtype)
            else:
                filter_torch = filter.to(device=img.device, dtype=img.dtype)
            img_f = torch.fft.fft2(img)
            img_filtered = torch.fft.ifft2(
                torch.fft.ifftshift(filter_torch, dim=(-2,-1)) * img_f
            ).real
        return img_filtered

def threshold_func(s_cov_set, fourier_angle=True, axis='all', two_fields = False):
    
    # Initialize the angle operator for the Fourier transform over angles
    angle_operator = FourierAngle()
    angle_operator_cross = FourierAngleCross()

    # Define the harmonic transform function with the modified mask
    def harmonic_transform(s_cov_set):
        # Get coefficient vectors and the index vectors
        coef = s_cov_set['for_synthesis']
        if not two_fields:
            idx = scale_annotation_a_b(to_numpy(s_cov_set['index_for_synthesis']).T)
            coef, idx = angle_operator(coef, idx, axis=axis)
        else:
            idx = scale_annotation_a_b_2fields(to_numpy(s_cov_set['index_for_synthesis']).T)
            coef, idx = angle_operator_cross(coef, idx, axis=axis)

        # Create a mask of the same length as the number of columns in coef
        mask = torch.zeros((coef.shape[-1],), dtype=torch.bool)

        # Always include non-Fourier types (mean, P00, S1) in the mask
        non_fourier_mask = np.isin(idx[:, 0], ['mean', 'P00', 'S1'])
        mask[torch.from_numpy(np.where(non_fourier_mask)[0])] = True

        # Filter only for C01 and C11 types (both real and imaginary)
        is_c01_c11 = np.isin(idx[:, 0], ['C01re', 'C01im', 'C11re', 'C11im'])

        # Extract relevant angular indices (l1, l2, l3) for these types
        l1 = idx[is_c01_c11, 4].astype(int)
        l2 = idx[is_c01_c11, 5].astype(int)
        l3 = idx[is_c01_c11, 6].astype(int)

        # Find valid indices where l1, l2, l3 are in {0, 1}
        valid_indices = (l1 <= 1) & (l2 <= 1) & (l3 <= 1)

        # Get the positions of valid coefficients
        valid_positions = np.where(is_c01_c11)[0][valid_indices]

        # Set the mask to True for these valid positions
        mask[torch.tensor(valid_positions, dtype=torch.long)] = True 

        # Output the transformed coefficients with the valid mask
        return coef[:, mask] if mask is not None else coef

    # Generate the threshold function that keeps only the first two harmonics
    threshold_func = lambda s_cov_set: harmonic_transform(s_cov_set)

    # Return the threshold function
    return threshold_func
