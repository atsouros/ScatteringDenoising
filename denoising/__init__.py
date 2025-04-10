#TEST FILE
import os
dirpath = os.path.dirname(__file__)

import numpy as np
from pathlib import Path
import time
import torch
import matplotlib.pyplot as plt
import sys
import camb

from denoising.utils import to_numpy
from denoising.FiltersSet import FiltersSet
from denoising.Scattering2d import Scattering2d
from denoising.polyspectra_calculators import get_power_spectrum, Bispectrum_Calculator, Trispectrum_Calculator
from denoising.AlphaScattering2d_cov import AlphaScattering2d_cov
from denoising.angle_transforms import FourierAngle
from denoising.scale_transforms import FourierScale


def compute_std_double(
    image1, image2, contamination_arr, image_ref1=None, image_ref2=None,
    J=None, L=4, M=None, N=None, l_oversampling=1, frequency_factor=1,
    mode='image',
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
    if mode=='image':
        _, M, N = image1.shape
    
    J = int(np.log2(min(M,N))) - 1        

    st_calc = Scattering2d(M, N, J, L, device, wavelets, l_oversampling=l_oversampling, frequency_factor=frequency_factor)
    def func(map1, ref_map1, map2=None, ref_map2=None):
        coef_list = []

        # Two-field case
        st_calc.add_ref_ab(ref_a=ref_map1, ref_b=ref_map2)

        def func_s(x1, x2):
            result = st_calc.scattering_cov_2fields(
                x1, x2, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria,
                normalization=normalization, remove_edge=remove_edge
            )
            select = ~torch.isin(result['index_for_synthesis'][0], torch.tensor([1, 3, 7, 11, 15, 19]))
            return result['for_synthesis'][:, select]
        
        coef_list.append(func_s(map1, map2))

        return torch.cat(coef_list, axis=-1)

    def std_func_dual(x1, ref1, x2, ref2, Mn=10, batch_size=5):

        # Convert inputs to torch tensors if needed
        x1 = torch.from_numpy(x1) if isinstance(x1, np.ndarray) else x1
        x2 = torch.from_numpy(x2) if isinstance(x2, np.ndarray) else x2
        ref1 = torch.from_numpy(ref1) if isinstance(ref1, np.ndarray) else ref1
        ref2 = torch.from_numpy(ref2) if isinstance(ref2, np.ndarray) else ref2

        # Set dtype and device
        dtype = torch.double if precision == 'double' else torch.float
        x1 = x1.to(device=device, dtype=dtype)
        x2 = x2.to(device=device, dtype=dtype)
        ref1 = ref1.to(device)
        ref2 = ref2.to(device)

        # Validate contamination_arr
        contamination_tensor = torch.from_numpy(contamination_arr).to(device=device, dtype=dtype)

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
    
    return std_func_dual(image1, image_ref1, image2, image_ref2)


def compute_std(
    target, contamination_arr,
    J=None, L=4, M=None, N=None, l_oversampling=1, frequency_factor=1, s_cov_func = None,
    mode='image',
    device='gpu', wavelets='morlet', seed=None,
    if_large_batch=False,
    C11_criteria=None,
    normalization='P00',
    precision='single', ps_bins=None, ps_bin_type='log', bispectrum_bins=None, bispectrum_bin_type='log',
    pseudo_coef=1,
    remove_edge=False
    ):

    '''
the estimator_name can be 's_mean', 's_mean_iso', 's_cov', 's_cov_iso', 'alpha_cov', 
the C11_criteria is the condition on j1 and j2 to compute coefficients, in addition to the condition that j2 >= j1. 
Use * or + to connect more than one condition.
    '''
    if not torch.cuda.is_available(): device='cpu'
    np.random.seed(seed)
    if C11_criteria is None:
        C11_criteria = 'j2>=j1'
    if mode=='image':
        _, M, N = target.shape
        
    if J is None:
        J = int(np.log2(min(M,N))) - 1
    
    # define calculator and estimator function
    st_calc = Scattering2d(M, N, J, L, device, wavelets, l_oversampling=l_oversampling, frequency_factor=frequency_factor)
    st_calc.add_ref(ref=target)

    func_s = lambda x: st_calc.scattering_cov(
        x, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria, 
        normalization=normalization, pseudo_coef=pseudo_coef,remove_edge=remove_edge)['for_synthesis']
            
    if s_cov_func is None:
        def func_s(x):
            return st_calc.scattering_cov(
                x, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria, 
                normalization=normalization, pseudo_coef=pseudo_coef, remove_edge=remove_edge
            )['for_synthesis']
    else:
        def func_s(x):
            coeffs =  st_calc.scattering_cov(
                x, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria, 
                normalization=normalization, pseudo_coef=pseudo_coef, remove_edge=remove_edge
            )
            return s_cov_func(coeffs)

    def func(image):
        coef_list = []
        coef_list.append(func_s(image))        
        return torch.cat(coef_list, axis=-1)
                
    def std_func(x, Mn=10, batch_size=5):
        # Convert to torch if necessary
        x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x

        # Set precision and device
        dtype = torch.double if precision == 'double' else torch.float
        x = x.to(device=device, dtype=dtype)

        contamination_tensor = torch.from_numpy(contamination_arr).to(device=device, dtype=dtype)

        # Compute reference statistics Φ(x)
        coeffs_ref = func(x).squeeze(0)  # Shape: (N_coeffs,)
        coeffs_number = coeffs_ref.size(0)

        # Prepare batches
        batch_number = (Mn + batch_size - 1) // batch_size
        COEFFS = torch.zeros((Mn, coeffs_number), device=device)

        for i in range(batch_number):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, Mn)

            # Slice batch of contamination
            cont_batch = contamination_tensor[start_idx:end_idx]  # Shape: (B, 1, H, W)

            # Broadcast and add to x
            cont_images = x.unsqueeze(0) + cont_batch  # Shape: (B, 1, H, W)

            for j in range(cont_images.size(0)):
                idx = start_idx + j
                COEFFS[idx] = func(cont_images[j]).squeeze(0)

        std_dev = COEFFS.std(dim=0, unbiased=False)
        return std_dev
    
    return std_func(target)

def denoise_double(
    target1, target2, contamination_arr, std, std_double = None, image_init1=None,image_init2=None, n_batch = 10, s_cov_func = None,
    J=None, L=4, M=None, N=None, l_oversampling=1, frequency_factor=1,
    mode='image', optim_algorithm='LBFGS', steps=300, learning_rate=0.2,
    device='gpu', wavelets='morlet', seed=None,
    if_large_batch=False,
    C11_criteria=None,
    normalization='P00',
    precision='single',
    print_each_step=False,
    Fourier=False, ps_bins=None, ps_bin_type='log',
    bi=False, bispectrum_bins=None, bispectrum_bin_type='log',
    ensemble=False,
    N_ensemble=1,
    pseudo_coef=1,
    remove_edge=False
):
    '''
the estimator_name can be 's_mean', 's_mean_iso', 's_cov', 's_cov_iso', 'alpha_cov', 
the C11_criteria is the condition on j1 and j2 to compute coefficients, in addition to the condition that j2 >= j1. 
Use * or + to connect more than one condition.
    '''
    if not torch.cuda.is_available(): device='cpu'
    np.random.seed(seed)
    if C11_criteria is None:
        C11_criteria = 'j2>=j1'
    if mode=='image':
        _, M, N = target1.shape
    
    # set initial point of synthesis
    if image_init1 is None and image_init2 is None:
        if mode=='image':
            if not ensemble:
                image_init1 = np.random.normal(
                    target1.mean((-2,-1))[:,None,None],
                    target1.std((-2,-1))[:,None,None],
                    (target1.shape[0], M, N)
                )
                image_init2 = np.random.normal(
                    target2.mean((-2,-1))[:,None,None],
                    target2.std((-2,-1))[:,None,None],
                    (target2.shape[0], M, N)
                )
                
            else:
                image_init1 = np.random.normal(
                    target1.mean(),
                    target1.std(),
                    (N_ensemble, M, N)
                )
                image_init2 = np.random.normal(
                    target2.mean(),
                    target2.std(),
                    (N_ensemble, M, N)
                )
        else:
            if M is None:
                print('please assign image size M and N.')
            if not ensemble: 
                image_init1 = np.random.normal(0,1,(target1.shape[0],M,N))
                image_init2 = np.random.normal(0,1,(target2.shape[0],M,N))
            else: 
                image_init1 = np.random.normal(0,1,(N_ensemble,M,N))
                image_init2 = np.random.normal(0,1,(N_ensemble,M,N))
        
    if J is None:
        J = int(np.log2(min(M,N))) - 1
    
    # power spectrum
    if ps_bins is None:
        ps_bins = J-1
    def func_ps(image):
        ps, _ = get_power_spectrum(image, bins=ps_bins, bin_type=ps_bin_type)
        return torch.cat(( (image.mean((-2,-1))/image.std((-2,-1)))[:,None], image.var((-2,-1))[:,None], ps), axis=-1)
    # bispectrum
    if bi:
        if bispectrum_bins is None:
            bispectrum_bins = J-1
        bi_calc = Bispectrum_Calculator(M, N, bins=bispectrum_bins, bin_type=bispectrum_bin_type, device=device)
        def func_bi(image):
            bi = bi_calc.forward(image)
            ps, _ = get_power_spectrum(image, bins=bispectrum_bins, bin_type=bispectrum_bin_type)
            return torch.cat(((image.mean((-2,-1))/image.std((-2,-1)))[:,None], ps, bi), axis=-1)

    st_calc = Scattering2d(M, N, J, L, device, wavelets, l_oversampling=l_oversampling, frequency_factor=frequency_factor)

    def func(map1, ref_map1, map2=None, ref_map2=None):
        coef_list = []

        if map2 is None:
            # Single-field case
            st_calc.add_ref(ref=ref_map1)

            if s_cov_func is None:
                def func_s(x):
                    return st_calc.scattering_cov(
                        x, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria, 
                        normalization=normalization, pseudo_coef=pseudo_coef, remove_edge=remove_edge
                    )['for_synthesis']
            else:
                def func_s(x):
                    coeffs =  st_calc.scattering_cov(
                        x, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria, 
                        normalization=normalization, pseudo_coef=pseudo_coef, remove_edge=remove_edge
                    )
                    return s_cov_func(coeffs)
            
            coef_list.append(func_s(map1))

        else:
            # Two-field case
            st_calc.add_ref_ab(ref_a=ref_map1, ref_b=ref_map2)

            def func_s(x1, x2):
                result = st_calc.scattering_cov_2fields(
                    x1, x2, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria,
                    normalization=normalization, remove_edge=remove_edge
                )
                select = ~torch.isin(result['index_for_synthesis'][0], torch.tensor([1, 3, 7, 11, 15, 19]))
                return result['for_synthesis'][:, select]
            
            coef_list.append(func_s(map1, map2))

        return torch.cat(coef_list, axis=-1)

    def BR_loss(target1, target2, image1, image2):
        loss1 = BR_loss_single(target1, image1, std[0], contamination_arr[:, 0])
        loss2 = BR_loss_single(target2, image2, std[1], contamination_arr[:, 1])
        loss3 = BR_loss_double(target1, image1, target2, image2, contamination_arr)
        return  (loss1 + loss2 + loss3) / 3
        
    def BR_loss_single(target, image, _std, contamination_arr):

        indices = np.random.choice(contamination_arr.shape[0], size=n_batch, replace=False)
        contamination_arr = contamination_arr[indices]
        # Convert to torch if needed
        if isinstance(contamination_arr, np.ndarray):
            contamination_arr = torch.from_numpy(contamination_arr)

        n_realizations = contamination_arr.shape[0]

        # Move contamination to correct device/dtype
        dtype = torch.double if precision == 'double' else torch.float
        contamination_tensor = contamination_arr.to(device=image.device, dtype=dtype)

        # Step 1: Compute reference statistics
        target_stats = func(target, target).squeeze(0)  # Shape: (N_coeffs,)

        # Step 2: Add contamination
        cont_images = image.unsqueeze(0) + contamination_tensor  # (n_realizations, 1, H, W)

        # Step 3: Compute noisy statistics
        noisy_stats_tensor = torch.stack(
            [func(cont_images[i], target) for i in range(n_realizations)],
            dim=0
        )  # Shape: (n_realizations, N_coeffs)

        # Step 4: Normalize and compute squared norm
        diff = noisy_stats_tensor - target_stats[None, :]
        normalized_diff = diff / _std[None, :]
        squared_norms = torch.sum(normalized_diff ** 2, dim=-1) / normalized_diff.size(-1)

        return squared_norms.mean()
    
    def BR_loss_double(target1, image1, target2, image2, contamination_arr):
        """
        Computes the BR loss for dual-input using precomputed contamination.

        Parameters
        ----------
        target1, image1 : torch.Tensor
            First frequency target and input (shape: (1, H, W)).
        target2, image2 : torch.Tensor
            Second frequency target and input (shape: (1, H, W)).
        contamination_arr : np.ndarray or torch.Tensor
            Contamination array of shape (n_realizations, 2, 1, H, W).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """

        # Randomly choose 10 unique indices from the first dimension
        indices = np.random.choice(contamination_arr.shape[0], size=n_batch, replace=False)
        contamination_arr = contamination_arr[indices]

        # Convert if needed
        if isinstance(contamination_arr, np.ndarray):
            contamination_arr = torch.from_numpy(contamination_arr)

        n_realizations = contamination_arr.shape[0]

        # Move to device with correct dtype
        dtype = torch.double if precision == 'double' else torch.float
        contamination_tensor = contamination_arr.to(device=image1.device, dtype=dtype)

        # Split contamination per channel
        cont1 = contamination_tensor[:, 0]  # (n_realizations, 1, H, W)
        cont2 = contamination_tensor[:, 1]  # (n_realizations, 1, H, W)

        # Step 1: Reference statistics
        target_stats = func(target1, target1, target2, target2).squeeze(0)  # (N_coeffs,)

        # Step 2: Add contamination
        cont_images1 = image1.unsqueeze(0) + cont1
        cont_images2 = image2.unsqueeze(0) + cont2

        # Step 3: Compute stats
        noisy_stats_tensor = torch.stack([
            func(cont_images1[i], target1, cont_images2[i], target2)
            for i in range(n_realizations)
        ], dim=0)  # (n_realizations, N_coeffs)

        # Step 4: Normalize and compute loss
        diff = noisy_stats_tensor - target_stats[None, :]
        normalized_diff = diff / std_double[None, :]
        squared_norms = torch.sum(normalized_diff ** 2, dim=-1) / normalized_diff.size(-1)

        return squared_norms.mean()    

    image_syn = denoise_general_double(
    target1, target2, image_init1, image_init2, func, BR_loss, 
    mode=mode, 
    optim_algorithm=optim_algorithm, steps=steps, learning_rate=learning_rate,
    device=device, precision=precision, print_each_step=print_each_step,
    Fourier=Fourier, ensemble=ensemble,
    )

    return image_syn

def denoise_general_double(
    target1, target2, image_init1, image_init2, estimator_function, loss_function, 
    mode='image', optim_algorithm='LBFGS', steps=100, learning_rate=0.5,
    device='gpu', precision='single', print_each_step=False, Fourier=False,
    ensemble=False,
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
    
    target1 = to_tensor(target1)
    target2 = to_tensor(target2)
    image_init1 = to_tensor(image_init1)
    image_init2 = to_tensor(image_init2)

    # calculate statistics for target images
    estimator_single = estimator_function(target1, target1)
    estimator_double = estimator_function(target1, target1, target2, target2)

    print('# of estimators: ', estimator_single.shape[-1] + estimator_double.shape[-1])
    
    # Define optimizable image model
    class OptimizableImage(torch.nn.Module):
        def __init__(self, input_init1, input_init2, Fourier=False):
            super().__init__()
            self.param1 = torch.nn.Parameter(input_init1)
            self.param2 = torch.nn.Parameter(input_init2)
            
            if Fourier: 
                self.image1 = torch.fft.ifftn(
                    self.param1[0] + 1j * self.param1[1],
                    dim=(-2, -1)).real
                self.image2 = torch.fft.ifftn(
                    self.param2[0] + 1j * self.param2[1],
                    dim=(-2, -1)).real
            else: 
                self.image1 = self.param1
                self.image2 = self.param2
    
    # Prepare input initialization for Fourier or direct space
    if Fourier: 
        temp1 = torch.fft.fftn(image_init1, dim=(-2, -1))
        temp2 = torch.fft.fftn(image_init2, dim=(-2, -1))
        input_init1 = torch.cat((temp1.real[None, ...], temp1.imag[None, ...]), dim=0)
        input_init2 = torch.cat((temp2.real[None, ...], temp2.imag[None, ...]), dim=0)
    else:
        input_init1 = torch.from_numpy(image_init1) if isinstance(image_init1, np.ndarray) else image_init1
        input_init2 = torch.from_numpy(image_init2) if isinstance(image_init2, np.ndarray) else image_init2

    # Ensure inputs are on the correct device and with the correct precision
    for var_name, var in [('input_init1', input_init1), ('input_init2', input_init2)]:
        if precision == 'double':
            var = var.type(torch.DoubleTensor)
        else:
            var = var.type(torch.FloatTensor)
        if device == 'gpu':
            var = var.cuda()
        globals()[var_name] = var

    image_model = OptimizableImage(input_init1, input_init2, Fourier)
        
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
        synthesized_image1 = image_model.image1
        synthesized_image2 = image_model.image2

        # Compute the estimator function on the synthesized images as a pair
        # Compute the loss using the loss function with the correct inputs
        loss = 0

        loss = loss_function(
            target1, target2, synthesized_image1, synthesized_image2
        )  # Regular case

        # Check for NaN loss
        if torch.isnan(loss):
            raise RuntimeError("Loss is NaN! Terminating process..")
        
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
    return (
        image_model.image1.cpu().detach().numpy(),
        image_model.image2.cpu().detach().numpy()
    )

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


def threshold_func_test(s_cov_set, fourier_angle=True, axis='all'):
    
    # Initialize the angle operator for the Fourier transform over angles
    angle_operator = FourierAngle()

    # Define the harmonic transform function with the modified mask
    def harmonic_transform(s_cov_set):
        # Get coefficient vectors and the index vectors
        coef = s_cov_set['for_synthesis']
        idx = scale_annotation_a_b(to_numpy(s_cov_set['index_for_synthesis']).T)
        
        # print("\nBefore Thresholding:")
        # cov_types, counts = np.unique(idx[:, 0], return_counts=True)
        # for cov_type, count in zip(cov_types, counts):
        #     print(f"Type: {cov_type}, Count: {count}")

        # Perform Fourier transform on angle indexes (l1, l2, l3) if enabled
        coef, idx = angle_operator(coef, idx, axis=axis)

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

        # print("\nAfter Thresholding:")
        # idx_after = idx[torch.from_numpy(np.where(mask.numpy())[0])]
        # cov_types_after, counts_after = np.unique(idx_after[:, 0], return_counts=True)
        # for cov_type, count in zip(cov_types_after, counts_after):
        #     print(f"Type: {cov_type}, Count: {count}")

        # Output the transformed coefficients with the valid mask
        return coef[:, mask] if mask is not None else coef

    # Generate the threshold function that keeps only the first two harmonics
    threshold_func = lambda s_cov_set: harmonic_transform(s_cov_set)

    # Return the threshold function
    return threshold_func