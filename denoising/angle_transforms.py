"""
Transforms performed on angles l1, l1p, l2
"""

import numpy as np
import torch


# def fourier(N, k) -> np.ndarray: # TODO. Define an orthonormal Fourier basis.

def P00_ft_index(P00_f, L):
    P00_fp = torch.cat((P00_f[...,0:1].real+1j*P00_f[...,L:L+1].real, P00_f[...,1:L]), dim=-1)
    return P00_fp

def C01_ft_index(C01_f, L):
    C01_fp = torch.cat((
        torch.cat((C01_f[...,0:1,0:1].real+1j*C01_f[...,0:1,L:L+1].real, C01_f[...,0:1,1:L]), dim=-1),
        C01_f[...,1:L//2,0:L],
        torch.cat((C01_f[...,L//2:L//2+1,0:1].real+1j*C01_f[...,L//2:L//2+1,L:L+1].real, C01_f[...,L//2:L//2+1,1:L]), dim=-1),
        C01_f[...,L//2+1:,1:L+1].roll(1,-1),
    ), dim=-2)
    return C01_fp

def C11_ft_index(C11_f, L):
    C11_fp = torch.cat((
        torch.cat((
            torch.cat((
                C11_f[...,0:1,0:1,0:1].real+1j*C11_f[...,0:1,0:1,L:L+1].real, 
                C11_f[...,0:1,0:1,1:L]), dim=-1),
            C11_f[...,0:1,1:L//2,0:L],
            torch.cat((
                C11_f[...,0:1,L//2:L//2+1,0:1].real+1j*C11_f[...,0:1,L//2:L//2+1,L:L+1].real, 
                C11_f[...,0:1,L//2:L//2+1,1:L]), dim=-1),
            C11_f[...,0:1,L//2+1:,1:L+1].roll(1,-1),
        ), dim=-2),
        C11_f[...,1:L//2,:,0:L],
        torch.cat((
            torch.cat((
                C11_f[...,L//2:L//2+1,0:1,0:1].real+1j*C11_f[...,L//2:L//2+1,0:1,L:L+1].real, 
                C11_f[...,L//2:L//2+1,0:1,1:L]), dim=-1),
            C11_f[...,L//2:L//2+1,1:L//2,0:L],
            torch.cat((
                C11_f[...,L//2:L//2+1,L//2:L//2+1,0:1].real+1j*C11_f[...,L//2:L//2+1,L//2:L//2+1,L:L+1].real, 
                C11_f[...,L//2:L//2+1,L//2:L//2+1,1:L]), dim=-1),
            C11_f[...,L//2:L//2+1,L//2+1:,1:L+1].roll(1,-1),
        ), dim=-2),
        C11_f[...,L//2+1:,:,1:L+1].roll(1,-1),
    ), dim=-3)
    return C11_fp
    
class FourierAngle:
    """
    Perform a Fourier transform along angles l1, l1p, l2.
    """
    def __init__(self):
        self.F = None

    def __call__(self, s_cov, idx_info, if_isotropic=True, axis='all'):
        '''
        do an angular fourier transform on 
        axis = 'all' or 'l1'
        '''
        cov_type, j1, a, b, l1, l2, l3 = idx_info.T

        L = l3.max() + 1  # number of angles # TODO. Hack, should better infer the value of L

        # P00   = s_cov[:, cov_type == 'P00']  .reshape(len(s_cov), -1, L)
        # S1    = s_cov[:, cov_type == 'S1']   .reshape(len(s_cov), -1, L)
        C01re = s_cov[:, cov_type == 'C01re'].reshape(len(s_cov), -1, L, L)
        C01im = s_cov[:, cov_type == 'C01im'].reshape(len(s_cov), -1, L, L)
        C11re = s_cov[:, cov_type == 'C11re'].reshape(len(s_cov), -1, L, L, L)
        C11im = s_cov[:, cov_type == 'C11im'].reshape(len(s_cov), -1, L, L, L)
        # P00_f = torch.fft.fftn(P00, norm='ortho', dim=(-1))
        # P00_fp = P00_ft_index(P00_f, L//2)
        
        # S1_f = torch.fft.fftn(S1, norm='ortho', dim=(-1))
        # S1_fp = P00_ft_index(S1_f, L//2)
                
        C01_half = C01re + 1j * C01im
        C01_f = torch.fft.fftn(torch.cat((C01_half, C01_half.conj()), dim=-1), norm='ortho', dim=(-2,-1)) / 2**0.5
        C01_fp = C01_ft_index(C01_f, L)
        
        C11_half = C11re + 1j * C11im
        C11_f = torch.fft.fftn(torch.cat((C11_half, C11_half.conj()), dim=-1), norm='ortho', dim=(-3,-2,-1)) / 2**0.5
        C11_fp = C11_ft_index(C11_f, L)
           
        # idx_info for mean, P00, S1
        cov_no_fourier = s_cov[:, np.isin(cov_type, ['mean', 'P00', 'S1'])]
        idx_info_no_fourier = idx_info[np.isin(cov_type, ['mean', 'P00', 'S1']), :]
        # cov_no_fourier = s_cov[:, np.isin(cov_type, ['mean'])]
        # idx_info_no_fourier = idx_info[np.isin(cov_type, ['mean']), :]

        # # idx_info for P00
        # P00_f_flattened = torch.cat([P00_fp.real.reshape(len(s_cov), -1), P00_fp.imag.reshape(len(s_cov), -1)], dim=-1)
        # idx_info_P00 = idx_info[np.isin(cov_type, ['P00']), :]
        # # idx_info for S1
        # S1_f_flattened = torch.cat([S1_fp.real.reshape(len(s_cov), -1), S1_fp.imag.reshape(len(s_cov), -1)], dim=-1)
        # idx_info_S1 = idx_info[np.isin(cov_type, ['S1']), :]
        
        # idx_info for C01
        C01_f_flattened = torch.cat([C01_fp.real.reshape(len(s_cov), -1), C01_fp.imag.reshape(len(s_cov), -1)], dim=-1)
        idx_info_C01 = idx_info[np.isin(cov_type, ['C01re', 'C01im']), :]

        # idx_info for C11
        C11_f_flattened = torch.cat([C11_fp.real.reshape(len(s_cov), -1), C11_fp.imag.reshape(len(s_cov), -1)], dim=-1)
        idx_info_C11 = idx_info[np.isin(cov_type, ['C11re', 'C11im']), :]

        idx_info_f = np.concatenate([idx_info_no_fourier, 
#                                      idx_info_P00, idx_info_S1, 
                                     idx_info_C01, idx_info_C11])
        s_covs_f = torch.cat([cov_no_fourier, 
#                               P00_f_flattened, S1_f_flattened, 
                              C01_f_flattened, C11_f_flattened], dim=-1)

        return s_covs_f, idx_info_f

class FourierAngleCross:
    """
    Perform a Fourier transform along angles l1, l1p, l2 for scattering coefficients.
    Applies only to C01 and C11 terms; leaves others untouched.
    """
    def __init__(self):
        pass

    def __call__(self, s_cov, idx_info, if_isotropic=True, axis='all'):
        cov_type, j1, a, b, l1, l1p, l2 = idx_info.T
        L = int(l2.max()) + 1 if len(l2) > 0 else 1

        def safe_fft(x, dim):
            try:
                return torch.fft.fftn(x, dim=dim, norm='ortho')
            except RuntimeError:
                return torch.fft.fftn(x.cpu(), dim=dim, norm='ortho').to(x.device)

        def process_complex_tensor(real, imag, dims):
            if real.shape[1] == 0:
                shape = list(real.shape)
                shape[dims[-1]] = 0
                return real.new_zeros(*shape, dtype=torch.complex64)

            complex_tensor = real + 1j * imag
            concat_tensor = torch.cat((complex_tensor, complex_tensor.conj()), dim=dims[-1])
            fft_result = safe_fft(concat_tensor, dim=dims) / 2**0.5
            return fft_result.index_select(dims[-1], torch.arange(real.shape[dims[-1]], device=real.device))

        # Extract relevant coefficients
        C01re = s_cov[:, cov_type == 'C01re']
        C01im = s_cov[:, cov_type == 'C01im']
        C11re = s_cov[:, cov_type == 'C11re']
        C11im = s_cov[:, cov_type == 'C11im']

        # Reshape for FFT
        C01re = C01re.reshape(len(s_cov), -1, L, L)
        C01im = C01im.reshape(len(s_cov), -1, L, L)
        C11re = C11re.reshape(len(s_cov), -1, L, L, L)
        C11im = C11im.reshape(len(s_cov), -1, L, L, L)

        # FFTs (complex)
        C01_f = process_complex_tensor(C01re, C01im, dims=(-2, -1))
        C11_f = process_complex_tensor(C11re, C11im, dims=(-3, -2, -1))

        # Flatten
        C01_f_flat = torch.cat([C01_f.real.reshape(len(s_cov), -1), C01_f.imag.reshape(len(s_cov), -1)], dim=-1)
        C11_f_flat = torch.cat([C11_f.real.reshape(len(s_cov), -1), C11_f.imag.reshape(len(s_cov), -1)], dim=-1)

        # Keep untouched coefficients
        untouched_mask = np.isin(cov_type, ['mean', 'P00', 'S1'])
        untouched_cov = s_cov[:, untouched_mask]
        untouched_idx = idx_info[untouched_mask]

        # Update idx_info for transformed C01 and C11
        C01_idx = idx_info[np.isin(cov_type, ['C01re', 'C01im'])]
        C11_idx = idx_info[np.isin(cov_type, ['C11re', 'C11im'])]

        idx_out = np.concatenate([untouched_idx, C01_idx, C11_idx], axis=0)
        coef_out = torch.cat([untouched_cov, C01_f_flat, C11_f_flat], dim=-1)

        return coef_out, idx_out            