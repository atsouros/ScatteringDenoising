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

    if device == 'gpu':
            contamination_arr = torch.tensor(contamination_arr).cuda()

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
    st_calc = Scattering2d(M, N, J, L, device, wavelets, l_oversampling=l_oversampling, frequency_factor=frequency_factor)

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

        dtype = torch.double if precision == 'double' else torch.float

        std_list = []

        for i, x in enumerate(target_tuple):
            x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
            x = x.to(device=device_name, dtype=dtype)

            st_calc.add_ref(ref=x)


            cont_i = contamination_arr[:, i]  # Shape: (Mn, 1, H, W)

            # Compute reference statistics Φ(x)
            coeffs_ref = func(x).squeeze(0)  # Shape: (N_coeffs,)
            coeffs_number = coeffs_ref.size(0)

            # Prepare batches
            batch_number = (Mn + batch_size - 1) // batch_size
            COEFFS = torch.zeros((Mn, coeffs_number), device=device_name)

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

    if image_ref is None:
        image_ref = image  

    if s_cov_func is None: 
        def func_s(x1, x2):
                coeff_dict = st_calc.scattering_cov_2fields(
                    x1, x2, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria,
                    normalization=normalization, remove_edge=remove_edge
                )
                # select = ~torch.isin(result['index_for_synthesis'][0], torch.tensor([1, 3, 7, 11, 15, 19]))
                return coeff_dict['for_synthesis']#[:, select] 
    else:
        def func_s(x1, x2):
                coeff_dict = st_calc.scattering_cov_2fields(
                    x1, x2, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria,
                    normalization=normalization, remove_edge=remove_edge
                )
                return s_cov_func(coeff_dict)


    st_calc = Scattering2d(M, N, J, L, device, wavelets, l_oversampling=l_oversampling, frequency_factor=frequency_factor)
    def func(map1, ref_map1, map2=None, ref_map2=None):
        coef_list = []

        # Two-field case
        st_calc.add_ref_ab(ref_a=ref_map1, ref_b=ref_map2)
        
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
        device_torch = torch.device('cuda' if device == 'gpu' else 'cpu')
        x1 = x1.to(device=device_torch, dtype=dtype)
        x2 = x2.to(device=device_torch, dtype=dtype)
        ref1 = ref1.to(device=device_torch, dtype=dtype)
        ref2 = ref2.to(device=device_torch, dtype=dtype)
        contamination_tensor = torch.from_numpy(contamination_arr).to(device=device_torch, dtype=dtype)

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
    
    return std_func_dual(image[0], image_ref[0], image[1], image_ref[1])

def denoise(
    target, contamination_arr, std, std_double = None, image_init = None, n_batch = 10, s_cov_func = None, s_cov_func_2fields = None,
    J=None, L=4, M=None, N=None, l_oversampling=1, frequency_factor=1, optim_algorithm='LBFGS', steps=300, learning_rate=0.2,
    device='gpu', wavelets='morlet', seed=None,
    if_large_batch=False,
    C11_criteria=None,
    normalization='P00',
    precision='single',
    print_each_step=False,
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
        
    if isinstance(target, tuple):
        _, M, N = target[0].shape
    else:
        _, M, N = target.shape 
    
    # set initial point of synthesis
    if image_init is None:
        image_init = target
        
    if J is None:
        J = int(np.log2(min(M,N))) - 1
    
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
            if s_cov_func_2fields is None:
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
                    return s_cov_func_2fields(coeff_dict)
                 
            
            coef_list.append(func_s(map1, map2))

        return torch.cat(coef_list, axis=-1)
    
    def loss_func(*args):
        assert len(args) % 2 == 0, "Expecting equal number of targets and images"
        mid = len(args) // 2
        targets = args[:mid]
        images = args[mid:]

        loss1 = loss_func_single(targets[0], images[0], std[0], contamination_arr[:, 0])
        loss2 = loss_func_single(targets[1], images[1], std[1], contamination_arr[:, 1])
        loss3 = loss_func_double(targets[0], images[0], targets[1], images[1], contamination_arr)

        return (loss1 + loss2 + loss3) / 3

    # def loss_func(*args):
    #     *targets, image = args

    #     T_d = 10
    #     nu = (217, 353)

    #     # Compute correct (unnormalized) scaling factors
    #     f1 = image.new_tensor(MBB_factor(T_d, nu[0] * 1e9))
    #     f2 = image.new_tensor(MBB_factor(T_d, nu[1] * 1e9))

    #     # Scale the shared image to create frequency-specific versions
    #     images = (image * f1 * 1e20, image * f2 * 1e20)

    #     # Compute loss
    #     loss1 = loss_func_single(targets[0], images[0], std[0], contamination_arr[:, 0])
    #     loss2 = loss_func_single(targets[1], images[1], std[1], contamination_arr[:, 1])
    #     loss3 = loss_func_double(targets[0], images[0], targets[1], images[1], contamination_arr)

    #     return (loss1 + loss2 + loss3) / 3

    def loss_func_single(target, image, _std, contamination_arr):
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
    
    def loss_func_double(target1, image1, target2, image2, contamination_arr):
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

    # calculate statistics for target images
    estimator_single = estimator_function(*targets)
    estimator_double = estimator_function(*targets, *targets)
    
    print('# of estimators: ', estimator_single.shape[-1] + estimator_double.shape[-1])
    
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
            img_f = torch.fft.fft2(img)
            img_filtered = torch.fft.ifft2(
                torch.fft.ifftshift(filter, dim=(-2,-1)) * img_f
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
