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

def generate_cmb_map(nside, spectral_index=-2.035, amplitude=1.0, device='cpu'):
    """
    Generates a Gaussian random field (GRF) with a power spectrum P(k) ~ k^spectral_index,
    and scales its amplitude.

    Args:
        nside (int): Number of pixels along x,y-axis.
        spectral_index (float, optional): Spectral index of the power spectrum. Defaults to -2.035.
        amplitude (float, optional): Scaling factor for the field's amplitude.
        device (str, optional): Device to store the tensor ('cpu' or 'cuda').

    Returns:
        torch.Tensor: GRF of shape (nside, nside).
    """
    # Create k-space grid
    kx = torch.fft.fftfreq(nside, d=1.0, device=device) * nside
    ky = torch.fft.fftfreq(nside, d=1.0, device=device) * nside
    kx, ky = torch.meshgrid(kx, ky, indexing="ij")
    k = torch.sqrt(kx**2 + ky**2)

    # Avoid division by zero at k = 0
    k_min = 1.0  # Avoid singularity at k=0
    k = torch.where(k == 0, torch.tensor(k_min, device=device), k)

    # Power spectrum P(k) ~ k^spectral_index
    P_k = k**spectral_index

    # Generate random Gaussian field in Fourier space
    real_part = torch.randn(nside, nside, device=device)
    imag_part = torch.randn(nside, nside, device=device)
    noise = real_part + 1j * imag_part

    # Apply power spectrum scaling
    fourier_field = noise * torch.sqrt(P_k)

    # Inverse Fourier Transform to real space
    field = torch.fft.ifft2(fourier_field).real

    # Normalize and scale by amplitude
    field = (field - field.mean()) / field.std()  # Normalize to mean 0, std 1
    field *= amplitude  # Scale by desired amplitude

    return field

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
    J=None, L=4, M=None, N=None, l_oversampling=1, frequency_factor=1,
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
    target1, target2, contamination_arr, std, std_double = None, image_init1=None,image_init2=None, n_batch = 10,
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

            def func_s(x):
                return st_calc.scattering_cov(
                    x, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria, 
                    normalization=normalization, pseudo_coef=pseudo_coef, remove_edge=remove_edge
                )['for_synthesis']
            
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


# denoise
def denoise(
    estimator_name, target, std, variance, amplitude = 1., n_realizations = 10, spectral_index = -2.035, image_init=None, image_ref=None, image_b=None,
    J=None, L=4, M=None, N=None, l_oversampling=1, frequency_factor=1,
    mode='image', optim_algorithm='LBFGS', steps=300, learning_rate=0.2,
    device='gpu', wavelets='morlet', seed=None,
    if_large_batch=False,
    C11_criteria=None,
    normalization='P00',
    precision='single',
    print_each_step=False,
    s_cov_func=None,
    s_cov_func_params=[],
    Fourier=False,
    target_full=None,
    ps=False, ps_bins=None, ps_bin_type='log',
    bi=False, bispectrum_bins=None, bispectrum_bin_type='log',
    phi4=False, phi4_j=False, hist=False, hist_factor=1, hist_j=False, hist_j_factor=1,
    ensemble=False,
    N_ensemble=1,
    reference_P00=None,
    pseudo_coef=1,
    remove_edge=False,
    cmb=False
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
    
    # set initial point of synthesis
    if image_init is None:
        if mode=='image':
            if not ensemble:
                image_init = np.random.normal(
                    target.mean((-2,-1))[:,None,None],
                    target.std((-2,-1))[:,None,None],
                    (target.shape[0], M, N)
                )
            else:
                image_init = np.random.normal(
                    target.mean(),
                    target.std(),
                    (N_ensemble, M, N)
                )
        else:
            if M is None:
                print('please assign image size M and N.')
            # if 's_' in estimator_name:
            #     image_init = get_random_data(target[:,], target, M=M, N=N, N_image=target.shape[0], mode='func', seed=seed)
            if not ensemble: image_init = np.random.normal(0,1,(target.shape[0],M,N))
            else: image_init = np.random.normal(0,1,(N_ensemble,M,N))
    elif type(image_init) is str:
        if image_init=='random phase':
            image_init = get_random_data(target, seed=seed) # gaussian field
        
    if J is None:
        J = int(np.log2(min(M,N))) - 1
    
    # define calculator and estimator function
    if 's' in estimator_name:
        st_calc = Scattering2d(M, N, J, L, device, wavelets, l_oversampling=l_oversampling, frequency_factor=frequency_factor)
        if mode=='image':
            if '2fields' not in estimator_name: 
                st_calc.add_ref(ref=target)
            else: 
                if image_b is None: print('should provide a valid image_b.')
                else: st_calc.add_ref_ab(ref_a=target, ref_b=image_b)
            if ensemble:
                ref_P00_mean = st_calc.ref_scattering_cov['P00'].mean(0)[None,:,:]
                if 'iso' in estimator_name: ref_P00_mean = ref_P00_mean.mean(2)[:,:,None]
                st_calc.ref_scattering_cov['P00'] = ref_P00_mean
        if mode=='estimator':
            if image_ref is None:
                if target_full is None:
                    temp = target
                else:
                    temp = target_full
                if normalization=='P00': 
                    if reference_P00 is None: st_calc.add_synthesis_P00(s_cov=temp, if_iso='iso' in estimator_name)
                    else: st_calc.add_synthesis_P00(P00=reference_P00)
                else: st_calc.add_synthesis_P11(temp, 'iso' in estimator_name, C11_criteria)
            else:
                st_calc.add_ref(ref=image_ref)

        if estimator_name=='s_mean_iso':
            func_s = lambda x: st_calc.scattering_coef(x, flatten=True)['for_synthesis_iso']
        if estimator_name=='s_mean':
            func_s = lambda x: st_calc.scattering_coef(x, flatten=True)['for_synthesis']
        if 's_cov_func' in estimator_name:
            def func_s(image):
                s_cov_set = st_calc.scattering_cov(
                    image, use_ref=True, if_large_batch=if_large_batch, 
                    C11_criteria=C11_criteria, 
                    normalization=normalization, pseudo_coef=pseudo_coef, remove_edge=remove_edge
                )
                return s_cov_func(s_cov_set, s_cov_func_params)
        if estimator_name=='s_cov_iso_para_perp':
            def func_s(image):
                result = st_calc.scattering_cov(
                    image, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria, 
                    normalization=normalization, pseudo_coef=pseudo_coef,remove_edge=remove_edge
                )
                index_type, j1, l1, j2, l2, j3, l3 = result['index_for_synthesis_iso']
                select = (index_type<3) + ((l2==0) + (l2==L//2)) * ((l3==0) + (l3==L//2) + (l3==-1))
                return result['for_synthesis_iso'][:,select]
        if estimator_name=='s_cov_iso_iso':
            def func_s(image):
                result = st_calc.scattering_cov(
                    image, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria, 
                    normalization=normalization, pseudo_coef=pseudo_coef, remove_edge=remove_edge
                )
                coef = result['for_synthesis_iso']
                index_type, j1, l1, j2, l2, j3, l3 = result['index_for_synthesis_iso']
                return torch.cat((
                    (coef[:,index_type<3]),
                    (coef[:,index_type==3].reshape(-1,L).mean(-1).reshape(len(coef),-1)),
                    (coef[:,index_type==4].reshape(-1,L).mean(-1).reshape(len(coef),-1)),
                    (coef[:,index_type==5].reshape(-1,L,L).mean((-2,-1)).reshape(len(coef),-1)),
                    (coef[:,index_type==6].reshape(-1,L,L).mean((-2,-1)).reshape(len(coef),-1)),
                ), dim=-1)
        if estimator_name=='s_cov_iso':
            func_s = lambda x: st_calc.scattering_cov(
                x, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria, 
                normalization=normalization, pseudo_coef=pseudo_coef,remove_edge=remove_edge)['for_synthesis_iso']
        if estimator_name=='s_cov':
            func_s = lambda x: st_calc.scattering_cov(
                x, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria, 
                normalization=normalization, pseudo_coef=pseudo_coef,remove_edge=remove_edge)['for_synthesis']
        if estimator_name=='s_cov_2fields_iso':
            def func_s(image):
                result = st_calc.scattering_cov_2fields(
                    image, image_b, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria,
                    normalization=normalization, remove_edge=remove_edge
                )
                select =(result['index_for_synthesis_iso'][0]!=1) * (result['index_for_synthesis_iso'][0]!=3) *\
                        (result['index_for_synthesis_iso'][0]!=7) * (result['index_for_synthesis_iso'][0]!=11)*\
                        (result['index_for_synthesis_iso'][0]!=15)* (result['index_for_synthesis_iso'][0]!=19)
                return result['for_synthesis_iso'][:,select]
        if estimator_name=='s_cov_2fields':
            def func_s(image):
                result = st_calc.scattering_cov_2fields(
                    image, image_b, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria,
                    normalization=normalization, remove_edge=remove_edge
                )
                select =(result['index_for_synthesis'][0]!=1) * (result['index_for_synthesis'][0]!=3) *\
                        (result['index_for_synthesis'][0]!=7) * (result['index_for_synthesis'][0]!=11) *\
                        (result['index_for_synthesis'][0]!=15)* (result['index_for_synthesis'][0]!=19)
                return result['for_synthesis'][:,select]
    if 'alpha_cov' in estimator_name:
        aw_calc = AlphaScattering2d_cov(M, N, J, L, wavelets=wavelets, device=device)
        func_s = lambda x: aw_calc.forward(x)
    
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
        
    def func(image):
        coef_list = []
        if estimator_name!='':
            coef_list.append(func_s(image))
        if ps:
            coef_list.append(func_ps(image))
        if bi:
            coef_list.append(func_bi(image))
        if phi4:
            coef_list.append(func_phi4(image))
        if phi4_j:
            coef_list.append(func_phi4_j(image, J))
        if hist:
            coef_list.append(hist_factor*func_hist(image))
        if hist_j:
            coef_list.append(hist_j_factor*func_hist_j(image, J))
        return torch.cat(coef_list, axis=-1)
    
    def BR_loss_noise(target, image):
        # Step 1: Compute the target statistics Φ(d)
        target_stats = func(target)  # Shape: (N_coeffs,)

        # Step 2: Generate all noise realizations at once
        noise_batch = torch.randn((n_realizations, *image.shape), device=image.device) * torch.sqrt(torch.tensor(variance, device=image.device))
        cmb_batch = torch.cat(
            [generate_cmb_map(nside=image.shape[-1], amplitude=amplitude, spectral_index = spectral_index, device=image.device).unsqueeze(0).unsqueeze(0) 
            for _ in range(n_realizations)], dim=0
        )
        
        if cmb:
            cont_batch = noise_batch + cmb_batch
        else:
            cont_batch = noise_batch

        # Step 3: Compute statistics Φ(u + n) in a batched manner
        cont_images = target - image.unsqueeze(0) + cont_batch  # Shape: (n_realizations, H, W)
        
        # Compute statistics for all noise realizations
        noisy_stats_tensor = torch.stack([func(cont_images[i]) for i in range(n_realizations)], dim=0)  # Shape: (n_realizations, N_coeffs)

        # Step 4: Compute the difference with correct broadcasting
        diff = noisy_stats_tensor - target_stats[None, :]  # Broadcasting ensures subtraction for all realizations

        # Step 5: Normalize by the **precomputed standard deviation**
        normalized_diff = diff / std[None, :]  # Ensure proper broadcasting
        N_coeff = normalized_diff.size()[-1]
        # Step 6: Compute squared norm
        squared_norms = torch.sum(normalized_diff ** 2, dim=-1) / N_coeff  # Shape: (n_realizations,)

        # Step 7: Compute final loss (mean over noise realizations)
        loss_tot1 = squared_norms.mean()

        target_stats = func(image) 
            
        noisy_stats_tensor = torch.stack([func(cont_batch[i]) for i in range(n_realizations)], dim=0)  # Shape: (n_realizations, N_coeffs)
        stds = noisy_stats_tensor.std(dim=(0, 1))


        diff = noisy_stats_tensor - target_stats[None, :]  # Broadcasting ensures subtraction for all realizations

        normalized_diff = diff / stds[None, :]  # Ensure proper broadcasting
        squared_norms = torch.sum(normalized_diff ** 2, dim=-1) / N_coeff  # Shape: (n_realizations,)
        loss_tot2 = squared_norms.mean()

        return (loss_tot1 + loss_tot2) / 2
        
    def BR_loss(target, image):
        """
        Computes the BR loss based on scattering statistics, following the same loss structure 
        as `compute_loss_BR` but adapted for a different statistical operator (`func()`).

        Parameters
        ----------
        target : torch.Tensor
            The target image d.
        image : torch.Tensor
            The current running map u.
        n_realizations : int, optional
            Total number of noise realizations for computing Φ(u + n).

        Returns
        -------
        float
            The computed loss a la Bruno.
        """

        # Step 1: Compute the target statistics Φ(d)
        target_stats = func(target)  # Shape: (N_coeffs,)

        # Step 2: Generate all noise realizations at once
        noise_batch = torch.randn((n_realizations, *image.shape), device=image.device) * torch.sqrt(torch.tensor(variance, device=image.device))
        
        if cmb:
            cmb_batch = torch.cat(
            [generate_cmb_map(nside=image.shape[-1], amplitude=amplitude, spectral_index = spectral_index, device=image.device).unsqueeze(0).unsqueeze(0) 
            for _ in range(n_realizations)], dim=0
        )
            cont_batch = noise_batch + cmb_batch
        else:
            cont_batch = noise_batch
                
        # Step 3: Compute statistics Φ(u + n) in a batched manner
        cont_images = image.unsqueeze(0) + cont_batch  # Shape: (n_realizations, H, W)
        
        # Compute statistics for all noise realizations
        noisy_stats_tensor = torch.stack([func(cont_images[i]) for i in range(n_realizations)], dim=0)  # Shape: (n_realizations, N_coeffs)

        # Step 4: Compute the difference with correct broadcasting
        diff = noisy_stats_tensor - target_stats[None, :]  # Broadcasting ensures subtraction for all realizations

        # Step 5: Normalize by the **precomputed standard deviation**
        normalized_diff = diff / std[None, :]  # Ensure proper broadcasting
        N_coeff = normalized_diff.size()[-1]
        # Step 6: Compute squared norm
        squared_norms = torch.sum(normalized_diff ** 2, dim=-1) / N_coeff  # Shape: (n_realizations,)

        # Step 7: Compute final loss (mean over noise realizations)
        loss_tot = squared_norms.mean()

        print(torch.mean(target))
        print(torch.mean(image))
        print(torch.mean(std))
        print(variance)
        print(loss_tot.item())
        sys.exit()


        return loss_tot
    
    def BR_loss_2(target, image):
        """
        Computes the BR loss based on scattering statistics, following the same loss structure 
        as `compute_loss_BR` but adapted for a different statistical operator (`func()`).

        Parameters
        ----------
        target : torch.Tensor
            The target image d.
        image : torch.Tensor
            The current running map u.
        n_realizations : int, optional
            Total number of noise realizations for computing Φ(u + n).

        Returns
        -------
        float
            The computed loss a la Bruno.
        """

        # Step 1: Compute the target statistics Φ(d)
        target_stats = func(target)  # Shape: (N_coeffs,)

        # Step 2: Generate all noise realizations at once
        noise_batch = torch.randn((n_realizations, *image.shape), device=image.device) * torch.sqrt(torch.tensor(variance, device=image.device))
        
        if cmb:
            cmb_batch = torch.cat(
            [generate_cmb_map(nside=image.shape[-1], amplitude=amplitude, spectral_index = spectral_index, device=image.device).unsqueeze(0).unsqueeze(0) 
            for _ in range(n_realizations)], dim=0
        )
        
            cont_batch = noise_batch + cmb_batch
        else:
            cont_batch = noise_batch
                
        # Step 3: Compute statistics Φ(u + n) in a batched manner
        cont_images = image.unsqueeze(0) + cont_batch  # Shape: (n_realizations, H, W)
        
        # Compute statistics for all noise realizations
        noisy_stats_tensor = torch.stack([func(cont_images[i]) for i in range(n_realizations)], dim=0)  # Shape: (n_realizations, N_coeffs)

        # Step 4: Compute the difference with correct broadcasting
        diff = noisy_stats_tensor - target_stats[None, :]  # Broadcasting ensures subtraction for all realizations

        # Step 5: Normalize by the **precomputed standard deviation**
        normalized_diff = diff / std[None, :]  # Ensure proper broadcasting
        N_coeff = normalized_diff.size()[-1]
        # Step 6: Compute squared norm
        squared_norms = torch.sum(normalized_diff ** 2, dim=-1) / N_coeff  # Shape: (n_realizations,)

        # Step 7: Compute final loss (mean over noise realizations)
        loss_tot1 = squared_norms.mean()

        # Step 1: Compute the target statistics Φ(d - u)
        target_stats = func(target - image)  # Shape: (N_coeffs,)

        # Step 2: Generate all noise realizations at once
        noise_batch = torch.randn((n_realizations, *image.shape), device=image.device) * torch.sqrt(torch.tensor(variance, device=image.device))
                        
        # Step 3: Compute statistics Φ(u + n) in a batched manner
        
        # Compute statistics for all noise realizations
        noisy_stats_tensor = torch.stack([func(cont_batch[i]) for i in range(n_realizations)], dim=0)  # Shape: (n_realizations, N_coeffs)

        noise_mean = noisy_stats_tensor.mean(dim=(0))

        stds = noisy_stats_tensor.std(dim=(0, 1))

        # Step 4: Compute the difference with correct broadcasting
        diff = noisy_stats_tensor - noise_mean[None, :]  # Broadcasting ensures subtraction for all realizations

        # Step 5: Normalize by the **precomputed standard deviation**
        normalized_diff = diff / stds[None, :]  # Ensure proper broadcasting
        N_coeff = normalized_diff.size()[-1]
        # Step 6: Compute squared norms
        squared_norms = torch.sum(normalized_diff ** 2, dim=-1) / N_coeff  # Shape: (n_realizations,)

        # Step 7: Compute final loss (mean over noise realizations)
        loss_tot2 = squared_norms.mean()

        return (loss_tot2 + loss_tot1) / 2 #divide by the number of terms in the loss
    
            
    image_syn = denoise_general(
        target, image_init, func, BR_loss, 
        mode=mode, 
        optim_algorithm=optim_algorithm, steps=steps, learning_rate=learning_rate,
        device=device, precision=precision, print_each_step=print_each_step,
        Fourier=Fourier, ensemble=ensemble,
    )
    return image_syn

# manipulate output of flattened scattering_cov
def denoise_general(
    target, image_init, estimator_function, loss_function, 
    mode='image', optim_algorithm='LBFGS', steps=100, learning_rate=0.5,
    device='gpu', precision='single', print_each_step=False, Fourier=False,
    ensemble=False,
):
    
    # define parameters
    N_image = image_init.shape[0]
    M = image_init.shape[1]
    N = image_init.shape[2]
    
    # formating target and image_init (to tensor, to cuda)
    if type(target)==np.ndarray:
        target = torch.from_numpy(target)
    if type(image_init)==np.ndarray:
        image_init = torch.from_numpy(image_init)
    if precision=='double':
        target = target.type(torch.DoubleTensor)
        image_init = image_init.type(torch.DoubleTensor)
    else:
        target = target.type(torch.FloatTensor)
        image_init = image_init.type(torch.FloatTensor)
    if device=='gpu':
        target     = target.cuda()
        image_init = image_init.cuda()

    # calculate statistics for target images
    if mode=='image':
        estimator_target = estimator_function(target)
    if mode=='estimator':
        estimator_target = target
    print('# of estimators: ', estimator_target.shape[-1])
    
    # define optimizable image model
    class OptimizableImage(torch.nn.Module):
        def __init__(self, input_init, Fourier=False):
            # super(OptimizableImage, self).__init__()
            super().__init__()
            self.param = torch.nn.Parameter( input_init )
            
            if Fourier: 
                self.image = torch.fft.ifftn(
                    self.param[0] + 1j*self.param[1],
                    dim=(-2,-1)).real
            else: self.image = self.param
    
    if Fourier: 
        temp = torch.fft.fftn(image_init, dim=(-2,-1))
        input_init = torch.cat((temp.real[None,...], temp.imag[None,...]), dim=0)
    else: input_init = image_init
    image_model = OptimizableImage(input_init, Fourier)
        
    # define optimizer
    if optim_algorithm   =='Adam':
        optimizer = torch.optim.Adam(image_model.parameters(), lr=learning_rate)
    elif optim_algorithm =='NAdam':
        optimizer = torch.optim.NAdam(image_model.parameters(), lr=learning_rate)
    elif optim_algorithm =='SGD':
        optimizer = torch.optim.SGD(image_model.parameters(), lr=learning_rate)
    elif optim_algorithm =='Adamax':
        optimizer = torch.optim.Adamax(image_model.parameters(), lr=learning_rate)
    elif optim_algorithm =='LBFGS':
        optimizer = torch.optim.LBFGS(image_model.parameters(), lr=learning_rate, 
            max_iter=steps, max_eval=None, 
            tolerance_grad=1e-19, tolerance_change=1e-19, 
            history_size=min(steps//2, 150), line_search_fn=None
        )
    
    def closure():
        optimizer.zero_grad()

        # Retrieve the synthesized image
        synthesized_image = image_model.image

        # Compute the loss using BR_loss, which takes the target image and the synthesized image
        loss = 0
        if ensemble:
            loss = loss_function(target, synthesized_image.mean(0))  # Ensemble case
        else:
            loss = loss_function(target, synthesized_image)          # Regular case

        # Print progress if required 
        if torch.isnan(loss):
            raise RuntimeError("Loss is NaN! Terminating process..")
        if print_each_step:
            print(f'Current Loss: {loss.item():.2e}')


        # Backpropagate the loss
        loss.backward()
        return loss
    
    # optimize
    t_start = time.time()
    if optim_algorithm =='LBFGS':
        i=0
        optimizer.step(closure)
    else:
        for i in range(steps):
            # print('step: ', i)
            optimizer.step(closure)
    t_end = time.time()
    print('time used: ', t_end - t_start, 's')

    return image_model.image.cpu().detach().numpy()


# synthesis
def synthesis(
    estimator_name, target, image_init=None, image_ref=None, image_b=None,
    J=None, L=4, M=None, N=None, l_oversampling=1, frequency_factor=1,
    mode='image', optim_algorithm='LBFGS', steps=300, learning_rate=0.2,
    device='gpu', wavelets='morlet', seed=None,
    if_large_batch=False,
    C11_criteria=None,
    normalization='P00',
    precision='single',
    print_each_step=False,
    s_cov_func=None,
    s_cov_func_params=[],
    Fourier=False,
    target_full=None,
    ps=False, ps_bins=None, ps_bin_type='log',
    bi=False, bispectrum_bins=None, bispectrum_bin_type='log',
    phi4=False, phi4_j=False, hist=False, hist_factor=1, hist_j=False, hist_j_factor=1,
    ensemble=False,
    N_ensemble=1,
    reference_P00=None,
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
    
    # set initial point of synthesis
    if image_init is None:
        if mode=='image':
            if not ensemble:
                image_init = np.random.normal(
                    target.mean((-2,-1))[:,None,None],
                    target.std((-2,-1))[:,None,None],
                    (target.shape[0], M, N)
                )
            else:
                image_init = np.random.normal(
                    target.mean(),
                    target.std(),
                    (N_ensemble, M, N)
                )
        else:
            if M is None:
                print('please assign image size M and N.')
            # if 's_' in estimator_name:
            #     image_init = get_random_data(target[:,], target, M=M, N=N, N_image=target.shape[0], mode='func', seed=seed)
            if not ensemble: image_init = np.random.normal(0,1,(target.shape[0],M,N))
            else: image_init = np.random.normal(0,1,(N_ensemble,M,N))
    elif type(image_init) is str:
        if image_init=='random phase':
            image_init = get_random_data(target, seed=seed) # gaussian field
        
    if J is None:
        J = int(np.log2(min(M,N))) - 1
    
    # define calculator and estimator function
    if 's' in estimator_name:
        st_calc = Scattering2d(M, N, J, L, device, wavelets, l_oversampling=l_oversampling, frequency_factor=frequency_factor)
        if mode=='image':
            if '2fields' not in estimator_name: 
                st_calc.add_ref(ref=target)
            else: 
                if image_b is None: print('should provide a valid image_b.')
                else: st_calc.add_ref_ab(ref_a=target, ref_b=image_b)
            if ensemble:
                ref_P00_mean = st_calc.ref_scattering_cov['P00'].mean(0)[None,:,:]
                if 'iso' in estimator_name: ref_P00_mean = ref_P00_mean.mean(2)[:,:,None]
                st_calc.ref_scattering_cov['P00'] = ref_P00_mean
        if mode=='estimator':
            if image_ref is None:
                if target_full is None:
                    temp = target
                else:
                    temp = target_full
                if normalization=='P00': 
                    if reference_P00 is None: st_calc.add_synthesis_P00(s_cov=temp, if_iso='iso' in estimator_name)
                    else: st_calc.add_synthesis_P00(P00=reference_P00)
                else: st_calc.add_synthesis_P11(temp, 'iso' in estimator_name, C11_criteria)
            else:
                st_calc.add_ref(ref=image_ref)

        if estimator_name=='s_mean_iso':
            func_s = lambda x: st_calc.scattering_coef(x, flatten=True)['for_synthesis_iso']
        if estimator_name=='s_mean':
            func_s = lambda x: st_calc.scattering_coef(x, flatten=True)['for_synthesis']
        if 's_cov_func' in estimator_name:
            def func_s(image):
                s_cov_set = st_calc.scattering_cov(
                    image, use_ref=True, if_large_batch=if_large_batch, 
                    C11_criteria=C11_criteria, 
                    normalization=normalization, pseudo_coef=pseudo_coef, remove_edge=remove_edge
                )
                return s_cov_func(s_cov_set, s_cov_func_params)
        if estimator_name=='s_cov_iso_para_perp':
            def func_s(image):
                result = st_calc.scattering_cov(
                    image, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria, 
                    normalization=normalization, pseudo_coef=pseudo_coef,remove_edge=remove_edge
                )
                index_type, j1, l1, j2, l2, j3, l3 = result['index_for_synthesis_iso']
                select = (index_type<3) + ((l2==0) + (l2==L//2)) * ((l3==0) + (l3==L//2) + (l3==-1))
                return result['for_synthesis_iso'][:,select]
        if estimator_name=='s_cov_iso_iso':
            def func_s(image):
                result = st_calc.scattering_cov(
                    image, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria, 
                    normalization=normalization, pseudo_coef=pseudo_coef, remove_edge=remove_edge
                )
                coef = result['for_synthesis_iso']
                index_type, j1, l1, j2, l2, j3, l3 = result['index_for_synthesis_iso']
                return torch.cat((
                    (coef[:,index_type<3]),
                    (coef[:,index_type==3].reshape(-1,L).mean(-1).reshape(len(coef),-1)),
                    (coef[:,index_type==4].reshape(-1,L).mean(-1).reshape(len(coef),-1)),
                    (coef[:,index_type==5].reshape(-1,L,L).mean((-2,-1)).reshape(len(coef),-1)),
                    (coef[:,index_type==6].reshape(-1,L,L).mean((-2,-1)).reshape(len(coef),-1)),
                ), dim=-1)
        if estimator_name=='s_cov_iso':
            func_s = lambda x: st_calc.scattering_cov(
                x, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria, 
                normalization=normalization, pseudo_coef=pseudo_coef,remove_edge=remove_edge)['for_synthesis_iso']
        if estimator_name=='s_cov':
            func_s = lambda x: st_calc.scattering_cov(
                x, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria, 
                normalization=normalization, pseudo_coef=pseudo_coef,remove_edge=remove_edge)['for_synthesis']
        if estimator_name=='s_cov_2fields_iso':
            def func_s(image):
                result = st_calc.scattering_cov_2fields(
                    image, image_b, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria,
                    normalization=normalization, remove_edge=remove_edge
                )
                select =(result['index_for_synthesis_iso'][0]!=1) * (result['index_for_synthesis_iso'][0]!=3) *\
                        (result['index_for_synthesis_iso'][0]!=7) * (result['index_for_synthesis_iso'][0]!=11)*\
                        (result['index_for_synthesis_iso'][0]!=15)* (result['index_for_synthesis_iso'][0]!=19)
                return result['for_synthesis_iso'][:,select]
        if estimator_name=='s_cov_2fields':
            def func_s(image):
                result = st_calc.scattering_cov_2fields(
                    image, image_b, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria,
                    normalization=normalization, remove_edge=remove_edge
                )
                select =(result['index_for_synthesis'][0]!=1) * (result['index_for_synthesis'][0]!=3) *\
                        (result['index_for_synthesis'][0]!=7) * (result['index_for_synthesis'][0]!=11) *\
                        (result['index_for_synthesis'][0]!=15)* (result['index_for_synthesis'][0]!=19)
                return result['for_synthesis'][:,select]
    if 'alpha_cov' in estimator_name:
        aw_calc = AlphaScattering2d_cov(M, N, J, L, wavelets=wavelets, device=device)
        func_s = lambda x: aw_calc.forward(x)
    
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
        
    def func(image):
        coef_list = []
        if estimator_name!='':
            coef_list.append(func_s(image))
        if ps:
            coef_list.append(func_ps(image))
        if bi:
            coef_list.append(func_bi(image))
        if phi4:
            coef_list.append(func_phi4(image))
        if phi4_j:
            coef_list.append(func_phi4_j(image, J))
        if hist:
            coef_list.append(hist_factor*func_hist(image))
        if hist_j:
            coef_list.append(hist_j_factor*func_hist_j(image, J))
        return torch.cat(coef_list, axis=-1)
    
    # define loss function
    def quadratic_loss(target, model):
        return ((target - model)**2).mean()*1e8
    
    # synthesis
    image_syn = synthesis_general(
        target, image_init, func, quadratic_loss, 
        mode=mode, 
        optim_algorithm=optim_algorithm, steps=steps, learning_rate=learning_rate,
        device=device, precision=precision, print_each_step=print_each_step,
        Fourier=Fourier, ensemble=ensemble,
    )
    return image_syn

# histogram
def func_hist(image):
    flat_image = image.reshape(len(image),-1)
    return flat_image.sort(dim=-1).values.reshape(len(image),-1,image.shape[-2]).mean(-1) / flat_image.std(-1)[:,None]
def smooth(image, j):
    M, N = image.shape[-2:]
    X = torch.arange(M)[:,None]
    Y = torch.arange(N)[None,:]
    R2 = (X-M//2)**2 + (Y-N//2)**2
    weight_f = torch.fft.fftshift(torch.exp(-0.5 * R2 / (M//(2**j)//2)**2)).cuda()
    image_smoothed = torch.fft.ifftn(torch.fft.fftn(image, dim=(-2,-1)) * weight_f[None,:,:], dim=(-2,-1))
    return image_smoothed.real
def func_hist_j(image, J):
    cumsum_list = []
    flat_image = image.reshape(len(image),-1)
    cumsum_list.append(
        flat_image.sort(dim=-1).values.reshape(len(image),-1,image.shape[-2]).mean(-1) / flat_image.std(-1)[:,None]
    )
    for j in range(J):
        subsample_rate = int(max(2**(j-1), 1))
        smoothed_image = smooth(image, j)[:,::subsample_rate,::subsample_rate]
        flat_image = smoothed_image.reshape(len(image),-1)
        cumsum_list.append(
            flat_image.sort(dim=-1).values.reshape(len(image),-1,smoothed_image.shape[-2]).mean(-1) / flat_image.std(-1)[:,None]
        )
    return torch.cat((cumsum_list), dim=-1)

def func_phi4(image):
    return (image**4).mean((-2,-1))[...,None]

def func_phi4_j(image, J):
    cumsum_list = []
    cumsum_list.append(
        (image**4).mean((-2,-1))[...,None] / (image**2).mean((-2,-1))[...,None]**2
    )
    for j in range(J):
        subsample_rate = int(max(2**(j-1), 1))
        smoothed_image = smooth(image, j)[:,::subsample_rate,::subsample_rate]
        cumsum_list.append( (smoothed_image**4).mean((-2,-1))[...,None] / (smoothed_image**2).mean((-2,-1))[...,None]**2 )
    return torch.cat((cumsum_list), dim=-1)

# manipulate output of flattened scattering_cov
def synthesis_general(
    target, image_init, estimator_function, loss_function, 
    mode='image', optim_algorithm='LBFGS', steps=100, learning_rate=0.5,
    device='gpu', precision='single', print_each_step=False, Fourier=False,
    ensemble=False,
):
    # define parameters
    N_image = image_init.shape[0]
    M = image_init.shape[1]
    N = image_init.shape[2]
    
    # formating target and image_init (to tensor, to cuda)
    if type(target)==np.ndarray:
        target = torch.from_numpy(target)
    if type(image_init)==np.ndarray:
        image_init = torch.from_numpy(image_init)
    if precision=='double':
        target = target.type(torch.DoubleTensor)
        image_init = image_init.type(torch.DoubleTensor)
    else:
        target = target.type(torch.FloatTensor)
        image_init = image_init.type(torch.FloatTensor)
    if device=='gpu':
        target     = target.cuda()
        image_init = image_init.cuda()
    
    # calculate statistics for target images
    if mode=='image':
        estimator_target = estimator_function(target)
    if mode=='estimator':
        estimator_target = target
    print('# of estimators: ', estimator_target.shape[-1])
    
    # define optimizable image model
    class OptimizableImage(torch.nn.Module):
        def __init__(self, input_init, Fourier=False):
            # super(OptimizableImage, self).__init__()
            super().__init__()
            self.param = torch.nn.Parameter( input_init )
            
            if Fourier: 
                self.image = torch.fft.ifftn(
                    self.param[0] + 1j*self.param[1],
                    dim=(-2,-1)).real
            else: self.image = self.param
    
    if Fourier: 
        temp = torch.fft.fftn(image_init, dim=(-2,-1))
        input_init = torch.cat((temp.real[None,...], temp.imag[None,...]), dim=0)
    else: input_init = image_init
    image_model = OptimizableImage(input_init, Fourier)
        
    # define optimizer
    if optim_algorithm   =='Adam':
        optimizer = torch.optim.Adam(image_model.parameters(), lr=learning_rate)
    elif optim_algorithm =='NAdam':
        optimizer = torch.optim.NAdam(image_model.parameters(), lr=learning_rate)
    elif optim_algorithm =='SGD':
        optimizer = torch.optim.SGD(image_model.parameters(), lr=learning_rate)
    elif optim_algorithm =='Adamax':
        optimizer = torch.optim.Adamax(image_model.parameters(), lr=learning_rate)
    elif optim_algorithm =='LBFGS':
        optimizer = torch.optim.LBFGS(image_model.parameters(), lr=learning_rate, 
            max_iter=steps, max_eval=None, 
            tolerance_grad=1e-19, tolerance_change=1e-19, 
            history_size=min(steps//2, 150), line_search_fn=None
        )
    
    def closure():
        optimizer.zero_grad()
        loss = 0
        estimator_model = estimator_function(image_model.image)
        if ensemble: loss = loss_function(estimator_model.mean(0), estimator_target.mean(0))
        else: loss = loss_function(estimator_model, estimator_target)
        if print_each_step:
            if optim_algorithm=='LBFGS' or (optim_algorithm!='LBFGS' and (i%100==0 or i%100==-1)):
                if not ensemble:
                    print((estimator_model-estimator_target).abs().max())
                    print(
                        'max residual: ', 
                        np.max((estimator_model - estimator_target).abs().detach().cpu().numpy()),
                        ', mean residual: ', 
                        np.mean((estimator_model - estimator_target).abs().detach().cpu().numpy()),
                    )
                else:
                    print((estimator_model.mean(0)-estimator_target.mean(0)).abs().max())
                    print(
                        'max residual: ', 
                        np.max((estimator_model.mean(0) - estimator_target.mean(0)).abs().detach().cpu().numpy()),
                        ', mean residual: ', 
                        np.mean((estimator_model.mean(0) - estimator_target.mean(0)).abs().detach().cpu().numpy()),
                    )
        loss.backward()
        return loss
    
    # optimize
    t_start = time.time()
    if not ensemble:
        print(
            'max residual: ', 
            np.max((estimator_function(image_model.image) - estimator_target).abs().detach().cpu().numpy()),
            ', mean residual: ', 
            np.mean((estimator_function(image_model.image) - estimator_target).abs().detach().cpu().numpy()),
        )
    else:
        print(
            'max residual: ', 
            np.max((estimator_function(image_model.image).mean(0) - estimator_target.mean(0)).abs().detach().cpu().numpy()),
            ', mean residual: ', 
            np.mean((estimator_function(image_model.image).mean(0) - estimator_target.mean(0)).abs().detach().cpu().numpy()),
        )
    if optim_algorithm =='LBFGS':
        i=0
        optimizer.step(closure)
    else:
        for i in range(steps):
            # print('step: ', i)
            optimizer.step(closure)
    if not ensemble:
        print(
            'max residual: ', 
            np.max((estimator_function(image_model.image) - estimator_target).abs().detach().cpu().numpy()),
            ', mean residual: ', 
            np.mean((estimator_function(image_model.image) - estimator_target).abs().detach().cpu().numpy()),
        )
    else:
        print(
            'max residual: ', 
            np.max((estimator_function(image_model.image).mean(0) - estimator_target.mean(0)).abs().detach().cpu().numpy()),
            ', mean residual: ', 
            np.mean((estimator_function(image_model.image).mean(0) - estimator_target.mean(0)).abs().detach().cpu().numpy()),
        )
    t_end = time.time()
    print('time used: ', t_end - t_start, 's')

    return image_model.image.cpu().detach().numpy()

# image pre-processing
def binning2x2(image):
    return (image[...,::2,::2] + image[...,1::2,::2] + image[...,::2,1::2] + image[...,1::2,1::2])/4

def whiten(image, overall=False):
    if overall:
        return (image - image.mean()) / image.std()
    else:
        return (image - image.mean((-2,-1))[:,None,None]) / image.std((-2,-1))[:,None,None]

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
        img_f = torch.fft.fft2(img)
        img_filtered = torch.fft.ifft2(
            torch.fft.ifftshift(filter, dim=(-2,-1)) * img_f
        ).real
    return img_filtered

def remove_slope(images):
    '''
        Removing the overall trend of an image by subtracting the result of
        a 2D linear fitting. This operation can reduce the edge effect when
        the field has too strong low-frequency components.
    '''
    M = images.shape[-2]
    N = images.shape[-1]
    z = images
    x = np.arange(M)[None,:,None]
    y = np.arange(N)[None,None,:]
    k_x = (
        (x - x.mean(-2)[:,None,:]) * (z - z.mean(-2)[:,None,:])
    ).mean((-2,-1)) / ((x - x.mean(-2)[:,None,:])**2).mean((-2,-1))
    k_y = (
        (y - y.mean(-1)[:,:,None]) * (z - z.mean(-1)[:,:,None])
    ).mean((-2,-1)) / ((y - y.mean(-1)[:,:,None])**2).mean((-2,-1))

    return z - k_x[:,None,None] * (x-M//2) - k_y[:,None,None] * (y-N//2)

# get derivatives for a vector field
def get_w_div(u):
    vxy = np.roll(u[...,0],-1,1) - np.roll(u[...,0],1,1)
    vxz = np.roll(u[...,0],-1,2) - np.roll(u[...,0],1,2)
    vyx = np.roll(u[...,1],-1,0) - np.roll(u[...,1],1,0)
    vyz = np.roll(u[...,1],-1,2) - np.roll(u[...,1],1,2)
    vzx = np.roll(u[...,2],-1,0) - np.roll(u[...,2],1,0)
    vzy = np.roll(u[...,2],-1,1) - np.roll(u[...,2],1,1)
    vxx = np.roll(u[...,0],-1,0) - np.roll(u[...,0],1,0)
    vyy = np.roll(u[...,1],-1,1) - np.roll(u[...,1],1,1)
    vzz = np.roll(u[...,2],-1,2) - np.roll(u[...,2],1,2)
    
    wx = vzy - vyz
    wy = vxz - vzx
    wz = vyx - vxy
    div = vxx + vyy + vzz
    return np.array([wx, wy, wz, div]).transpose((1,2,3,4,0))

# get random initialization
def get_random_data(target, M=None, N=None, N_image=None, mode='image', seed=None):
    '''
    get a gaussian random field with the same power spectrum as the image 'target' (in the 'image' mode),
    or with an assigned power spectrum function 'target' (in the 'func' mode).
    '''
    
    fftshift = np.fft.fftshift
    fft2 = np.fft.fft2
    ifft2 = np.fft.ifft2
    
    np.random.seed(seed)
    if mode == 'image':
        N_image = target.shape[0]
        M = target.shape[-2]
        N = target.shape[-1]
        random_phase       = np.random.rand(N_image, M//2-1,N-1)
        random_phase_left  = np.random.rand(N_image, M//2-1, 1)
        random_phase_top   = np.random.rand(N_image, 1, N//2-1)
        random_phase_middle= np.random.rand(N_image, 1, N//2-1)
        random_phase_corners=np.random.randint(0,2,(N_image, 3))/2
    if mode == 'func':
        random_phase       = np.random.normal(0,1,(N_image,M//2-1,N-1)) + np.random.normal(0,1,(N_image,M//2-1,N-1))*1j
        random_phase_left  = (np.random.normal(0,1,(N_image,M//2-1,1)) + np.random.normal(0,1,(N_image,M//2-1,1))*1j)
        random_phase_top   = (np.random.normal(0,1,(N_image,1,N//2-1)) + np.random.normal(0,1,(N_image,1,N//2-1))*1j)
        random_phase_middle= (np.random.normal(0,1,(N_image,1,N//2-1)) + np.random.normal(0,1,(N_image,1,N//2-1))*1j)
        random_phase_corners=np.random.normal(0,1,(N_image,3))
        
    gaussian_phase = np.concatenate((
        np.concatenate((
            random_phase_corners[:,1,None,None], random_phase_left, 
            random_phase_corners[:,2,None,None], -random_phase_left[:,::-1,:]
        ),axis=-2),
        np.concatenate((
            np.concatenate((random_phase_top, random_phase_corners[:,0,None,None], -random_phase_top[:,:,::-1]),axis=-1), random_phase,
            np.concatenate((random_phase_middle, np.zeros(N_image)[:,None,None], -random_phase_middle[:,:,::-1]),axis=-1), -random_phase[:,::-1,::-1],
        ),axis=-2),
    ),axis=-1)

    if mode == 'image':
        gaussian_modulus = np.abs(fftshift(fft2(target)))
        gaussian_field = ifft2(fftshift(gaussian_modulus*np.exp(1j*2*np.pi*gaussian_phase)))
    if mode == 'func':
        X = np.arange(0,M)
        Y = np.arange(0,N)
        Xgrid, Ygrid = np.meshgrid(X,Y, indexing='ij')
        R = ((Xgrid-M/2)**2+(Ygrid-N/2)**2)**0.5
        gaussian_modulus = target(R)
        gaussian_modulus[M//2, N//2] = 0
        gaussian_field = ifft2(fftshift(gaussian_modulus[None,:,:]*gaussian_phase))
    data = fftshift(gaussian_field.real)
    return np.ascontiguousarray(data)

# transforming scattering representation s_cov['for_synthesis_iso']
# angular fft of C01 and C11
def fft_coef(coef, index_type, L):
    C01_f = torch.fft.fft(
        coef[:,index_type==3].reshape(len(coef),-1,L) +
        coef[:,index_type==4].reshape(len(coef),-1,L) * 1j, norm='ortho')
    C11_f =torch.fft.fft2(
        coef[:,index_type==5].reshape(len(coef),-1,L,L) +
        coef[:,index_type==6].reshape(len(coef),-1,L,L) * 1j, norm='ortho')
    return C01_f, C11_f

def s_cov_WN(st_calc, N_realization=500, coef_name='for_synthesis_iso'):
    '''
    compute the mean and std of C01_f and C11_f for gaussian white noise.
    '''
    image_gaussian = np.random.normal(0,1,(50,st_calc.M,st_calc.N))
    s_cov = st_calc.scattering_cov(image_gaussian, if_large_batch=True)
    coef = s_cov[coef_name]
    index_type, j1, j2, j3, l1, l2, l3 = s_cov['index_'+coef_name]

    for i in range(N_realization//50-1):
        image_gaussian = np.random.normal(0,1,(50,st_calc.M,st_calc.N))
        s_cov  = st_calc.scattering_cov(image_gaussian, if_large_batch=True)
        coef = torch.cat((coef, s_cov[coef_name]), dim=0)
    C01_f, C11_f = fft_coef(coef, index_type, st_calc.L)
    return C01_f.mean(0), C01_f.std(0), C11_f.mean(0), C11_f.std(0)

# select s_cov['for_synthesis_iso'] with mask
def s_cov_iso_threshold(s_cov, param_list):
    '''
    this can be any function that eats the s_cov from denoising.s_cov()
    and some other parameters, and then outputs a flattened torch tensor.
    The output is flattened instead of with size of [N_image, -1] because
    the mask for each image can be different.
    '''
    L = param_list[0]
    coef = s_cov['for_synthesis_iso']
    index_type, j1, j2, j3, l1, l2, l3 = s_cov['index_for_synthesis_iso']
    
    # Fourier transform for l2-l1 and l3-l1
    C01_f, C11_f = fft_coef(coef, index_type, L)
    return torch.cat((
            coef[:,index_type<3].reshape(-1), # mean, P, S1
            (C01_f[param_list[1]].reshape(-1)).real,
            (C01_f[param_list[1]].reshape(-1)).imag,
            (C11_f[param_list[2]].reshape(-1)).real,
            (C11_f[param_list[2]].reshape(-1)).imag,
        ), dim=0)

def modify_angular(s_cov_set, factor, C01=False, C11=False, keep_para=False, ref_along='both'):
    '''
    a function to change the angular oscillation of C01 and/or C11 by a factor
    '''
    index_type, j1, j2, j3, l1, l2, l3 = s_cov_set['index_for_synthesis_iso']
    L = s_cov_set['P00'].shape[-1]
    s_cov = s_cov_set['for_synthesis_iso']*1.
    N_img = len(s_cov)
    if keep_para:
        if C01:
            s_cov[:,index_type==3] += (
                s_cov[:,index_type==3].reshape(N_img,-1,L) - 
                s_cov[:,index_type==3].reshape(N_img,-1,L)[:,:,0:1]
            ).reshape(N_img,-1) * factor
        if C11:
            if ref_along=='both':
                s_cov[:,index_type==5] += (
                    s_cov[:,index_type==5].reshape(N_img,-1,L,L) - 
                    s_cov[:,index_type==5].reshape(N_img,-1,L,L)[:,:,0:1,0:1]
                ).reshape(N_img,-1) * factor
            elif ref_along=='j12':
                s_cov[:,index_type==5] += (
                    s_cov[:,index_type==5].reshape(N_img,-1,L,L) - 
                    s_cov[:,index_type==5].reshape(N_img,-1,L,L)[:,:,0:1,:]
                ).reshape(N_img,-1) * factor
            elif ref_along=='j13':
                s_cov[:,index_type==5] += (
                    s_cov[:,index_type==5].reshape(N_img,-1,L) - 
                    s_cov[:,index_type==5].reshape(N_img,-1,L)[:,:,0:1]
                ).reshape(N_img,-1) * factor
    else:
        if C01:
            s_cov[:,index_type==3] += (
                s_cov[:,index_type==3].reshape(N_img,-1,L) - 
                s_cov[:,index_type==3].reshape(N_img,-1,L).mean(-1)[:,:,None]
            ).reshape(N_img,-1) * factor
        if C11:
            if ref_along=='both':
                s_cov[:,index_type==5] += (
                    s_cov[:,index_type==5].reshape(N_img,-1,L,L) - 
                    s_cov[:,index_type==5].reshape(N_img,-1,L,L).mean((-2,-1))[:,:,None,None]
                ).reshape(N_img,-1) * factor
            if ref_along=='j12':
                s_cov[:,index_type==5] += (
                    s_cov[:,index_type==5].reshape(N_img,-1,L,L) - 
                    s_cov[:,index_type==5].reshape(N_img,-1,L,L).mean((-2))[:,:,None,:]
                ).reshape(N_img,-1) * factor
            if ref_along=='j13':
                s_cov[:,index_type==5] += (
                    s_cov[:,index_type==5].reshape(N_img,-1,L,L) - 
                    s_cov[:,index_type==5].reshape(N_img,-1,L,L).mean((-1))[:,:,:,None]
                ).reshape(N_img,-1) * factor
    return s_cov

# show three panel plots
def show(image_target, image_syn, hist_range=(-2, 2), hist_bins=50, denoise = False):
    for i in range(len(image_target)):
        plt.figure(figsize=(9,3), dpi=200)
        plt.subplot(131) 
        plt.imshow(image_target[i], vmin=hist_range[0], vmax=hist_range[1])
        if not denoise:
            plt.xticks([]); plt.yticks([]); plt.title('original field')
        else:
            plt.xticks([]); plt.yticks([]); plt.title('data')
        plt.subplot(132)
        plt.imshow(image_syn[i], vmin=hist_range[0], vmax=hist_range[1])
        if not denoise:
            plt.xticks([]); plt.yticks([]); plt.title('synthesised field')
        else:
            plt.xticks([]); plt.yticks([]); plt.title('recovered field')
        plt.subplot(133); 
        plt.hist(image_target[i].flatten(), hist_bins, hist_range, histtype='step', label='target')
        plt.hist(   image_syn[i].flatten(), hist_bins, hist_range, histtype='step', label='synthesized')
        plt.yscale('log'); plt.legend(loc='lower center'); plt.title('histogram')
        plt.show()
        

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


if __name__ == "__main__":

    fourier_angle = True
    fourier_scale = True

    angle_operator = FourierAngle()
    scale_operator = FourierScale()

    def moments(s_cov, params):
        idx_info = to_numpy(s_cov['index_for_synthesis_iso']).T
        idx_info = scale_annotation_a_b(idx_info)
        s_cov = s_cov['for_synthesis_iso']

        if fourier_angle:
            s_cov, idx_info = angle_operator(s_cov, idx_info)
        if fourier_scale:
            s_cov, idx_info = scale_operator(s_cov, idx_info)

        return s_cov

    im1_path = Path(dirpath) / 'example_fields.npy'
    im = np.load(str(im1_path))

    image_syn = synthesis('s_cov_func', im[:1, :, :], s_cov_func=moments, J=7, steps=100, seed=0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), squeeze=False)
    axes[0, 0].imshow(im[0, :, :], cmap='viridis')
    axes[0, 0].grid(None)
    axes[0, 1].imshow(image_syn[0, :, :], cmap='viridis')
    axes[0, 1].grid(None)
    plt.show()

    print(0)
    
    
# for large data, scattering computation needs to be chunked to hold on memory
def chunk_model(X, st_calc, nchunks, **kwargs):
    partition = np.array_split(np.arange(X.shape[0]), nchunks)
    covs_l = [] 
    covs_l_iso = [] 
    for part in partition:
        X_here = X[part,:,:]
        
        s_cov_here = st_calc.scattering_cov(X_here, **kwargs)

        # keep relevent information only
        for key in list(s_cov_here.keys()):
            if key not in [
                'index_for_synthesis_iso', 'for_synthesis_iso',
                'index_for_synthesis', 'for_synthesis'
            ]:
                del s_cov_here[key]
            idx_iso = to_numpy(s_cov_here['index_for_synthesis_iso']).astype(object)
            cov_iso = s_cov_here['for_synthesis_iso']#.cpu()
            idx = to_numpy(s_cov_here['index_for_synthesis']).astype(object)
            cov = s_cov_here['for_synthesis']#.cpu()
        
        covs_l_iso.append(cov_iso)
        covs_l.append(cov)
    s_cov_set = {'index_for_synthesis_iso':idx_iso, 'for_synthesis_iso':torch.cat(covs_l_iso),
             'index_for_synthesis':idx, 'for_synthesis':torch.cat(covs_l)}
    return s_cov_set

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


def convolve_by_FFT(field, func_in_Fourier, device='cpu'):
    '''
    get the power spectrum of a given image
    '''
    if type(field) == np.ndarray:
        field = torch.from_numpy(field)
    M, N = field.shape[-2:]
    field_f = torch.fft.fftn(field, dim=(-2,-1))
    
    kx = torch.arange(0,M)
    ky = torch.arange(0,N)
    kx_grid, ky_grid = torch.meshgrid(ky, kx, indexing='ij')
    k_grid = torch.fft.ifftshift( ((kx_grid - M//2)**2 + (ky_grid - N//2)**2)**0.5, dim=(-2,-1))
    if device=='gpu':
      k_grid = k_grid.cuda()
      field_f = field_f.cuda()
    filter_f = func_in_Fourier( k_grid )

    filter_f_modi = filter_f + 0.
    filter_f_modi[0,0] = 0
    # print(filter_f.device, filter_f_modi.device)
    field_after_conv = torch.fft.ifftn( 
        field_f * filter_f_modi, dim=(-2,-1)
    ).real

    # filter_f[0,0] = 1

    return field_after_conv, filter_f, k_grid>0


# util to reduce ST coefficients
def reduced_ST(S, J, L):
    s0 = S[:,0:1]
    s1 = S[:,1:J+1]
    s2 = S[:,J+1:].reshape((-1,J,J,L))
    s21 = (s2.mean(-1) / s1[:,:,None]).reshape((-1,J**2))
    s22 = (s2[:,:,:,0] / s2[:,:,:,L//2]).reshape((-1,J**2))
    
    s1 = np.log(s1)
    select = s21[0]>0
    s21 = np.log(s21[:, select])
    s22 = np.log(s22[:, select])
    
    j1 = (np.arange(J)[:,None] + np.zeros(J)[None,:]).flatten()
    j2 = (np.arange(J)[None,:] + np.zeros(J)[:,None]).flatten()
    j1j2 = np.concatenate((j1[None, select], j2[None, select]), axis=0)
    return s0, s1, s21, s22, s2, j1j2
