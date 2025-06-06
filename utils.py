import numpy as np
import scipy.stats as stats
import torch
import torch.fft
from scipy.constants import h, k, c
from astropy import units as u
from scipy.ndimage import gaussian_filter


def preprocessing_euclid(field):
    alpha = 0.5
    epsilon = 1e-6
    
    field_shifted = field - np.min(field) + epsilon
    field_powered = np.power(field_shifted, alpha)
    field_powered = (field_powered - field_powered.mean()) / field_powered.std()

    return field_powered

def preprocessing_weak_lensing(x):
    L = 128
    N = x.shape[1] // L

    x = x.reshape((len(x), N, L, N, L)).swapaxes(2, 3).reshape((len(x) * N * N, L, L))
    x = preprocessing_euclid(x)
    return x

def power_spectrum(image):
    assert image.shape[0] == image.shape[1]    
    n = image.shape[0]

    fourier = np.fft.fftn(image)
    amplitude = (np.abs(fourier) ** 2).flatten()

    kfreq = np.fft.fftfreq(n) * n
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = (kfreq2D[0] ** 2 + kfreq2D[1] ** 2).flatten() ** (1 / 2)

    kbins = np.arange(1 / 2, n // 2 + 1, 1)
    kvals = (kbins[1:] + kbins[:-1]) / 2
    bins, _, _ = stats.binned_statistic(knrm, amplitude, statistic = "mean", bins = kbins)
    
    return kvals, bins

def gen_gaussian_field(n, spectrum):
    kfreq = np.fft.fftfreq(n) * n
    kfreq2D = np.meshgrid(kfreq, kfreq)

    knrm = (kfreq2D[0] ** 2 + kfreq2D[1] ** 2).flatten()[None, :] ** (1 / 2)
    kbins = np.arange(1 / 2, n // 2 + 1, 1)[:, None]

    mask = (knrm < kbins[1:]) & (knrm >= kbins[:-1])
    var = np.sum(mask * spectrum[:, None], axis=0) ** (1 / 2)
    fourier_field = var * np.random.randn(n * n) * np.exp(2 * np.pi * 1j * np.random.random(n * n))
    fourier_field = np.reshape(fourier_field, (n, n))
    field = np.fft.ifftn(fourier_field).real
    field = (field - np.mean(field)) / np.std(field)
    
    return field

def cross_spectrum(image1, image2):
    """
    Computes the cross spectrum of two images.
    
    Parameters
    ----------
    image1, image2 : np.ndarray
        Input images (must have the same shape).
    
    Returns
    -------
    kvals : np.ndarray
        The wave numbers (spatial frequencies).
    cross_spectrum : np.ndarray
        The average cross spectrum for each wave number bin.
    """

    assert image1.shape == image2.shape, "Input images must have the same shape."
    assert image1.shape[0] == image1.shape[1], "Input images must be square."
    
    n = image1.shape[0]

    # Compute the Fourier transforms of both images
    fourier1 = np.fft.fftn(image1)
    fourier2 = np.fft.fftn(image2)

    # Compute the cross power spectrum (complex multiplication)
    cross_amplitude = (fourier1 * np.conj(fourier2)).flatten()

    # Compute wave numbers
    kfreq = np.fft.fftfreq(n) * n
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2).flatten()

    # Define wave number bins and compute the binned cross spectrum
    kbins = np.arange(1 / 2, n // 2 + 1, 1)
    kvals = (kbins[1:] + kbins[:-1]) / 2
    cross_spectrum, _, _ = stats.binned_statistic(knrm, cross_amplitude.real, statistic="mean", bins=kbins)

    return kvals, cross_spectrum

def generate_cmb_map(n_x, n_y, spectral_index=-2.035, amplitude=3.0, device='cpu'):
    """
    Generates a Gaussian random field (GRF) with a power spectrum P(k) ~ k^spectral_index,
    and scales its amplitude.

    Args:
        n_x (int): Number of pixels along x-axis.
        n_y (int): Number of pixels along y-axis.
        spectral_index (float, optional): Spectral index of the power spectrum. Defaults to -2.035.
        amplitude (float, optional): Scaling factor for the field's amplitude.
        device (str, optional): Device to store the tensor ('cpu' or 'cuda').

    Returns:
        torch.Tensor: GRF of shape (n_x, n_y).
    """
    # Create k-space grid
    kx = torch.fft.fftfreq(n_x, d=1.0, device=device) * n_x
    ky = torch.fft.fftfreq(n_y, d=1.0, device=device) * n_y
    kx, ky = torch.meshgrid(kx, ky, indexing="ij")
    k = torch.sqrt(kx**2 + ky**2)

    # Avoid division by zero at k = 0
    k_min = 1.0  # Avoid singularity at k=0
    k = torch.where(k == 0, torch.tensor(k_min, device=device), k)

    # Power spectrum P(k) ~ k^spectral_index
    P_k = k**spectral_index

    # Generate random Gaussian field in Fourier space
    real_part = torch.randn(n_x, n_y, device=device)
    imag_part = torch.randn(n_x, n_y, device=device)
    noise = real_part + 1j * imag_part

    # Apply power spectrum scaling
    fourier_field = noise * torch.sqrt(P_k)

    # Inverse Fourier Transform to real space
    field = torch.fft.ifft2(fourier_field).real

    # Normalize and scale by amplitude
    field = (field - field.mean()) / field.std()  # Normalize to mean 0, std 1
    field *= amplitude  # Scale by desired amplitude

    return field

def modified_blackbody(N_H, T_d, nu, nu0=353e9, sigma_nu0=7.1e-27, beta=1.53):
    """
    Compute the mock observed dust intensity map using the modified blackbody spectrum.

    Parameters:
    - N_H : 2D numpy array
        Column density map (in cm^-2, log scale).
    - T_d : float
        Constant dust temperature (in K).
    - nu : float
        Observed frequency (in Hz).
    - nu0 : float, optional
        Reference frequency (default: 353 GHz).
    - sigma_nu0 : float, optional
        Mean dust opacity at reference frequency (default: 7.1e-27 cm^2 per H at 353 GHz, Planck XVII).
    - beta : float, optional
        Dust emissivity spectral index (default: 1.53, Planck XVII).

    Returns:
    - I_nu_μK : 2D numpy array
        Observed intensity map converted to μK_CMB.
    """
    # Compute frequency-dependent dust opacity
    sigma_nu = sigma_nu0 * (nu / nu0) ** beta

    # Compute the optical depth
    tau_nu = 10**N_H * sigma_nu

    # Compute the Planck function using numerically stable expm1
    x = h * nu / (k * T_d)
    B_nu = (2 * h * nu**3 / c**2) / np.expm1(x)

    # Compute intensity map in W·m^-2·Hz^-1·sr^-1
    I_nu = tau_nu * B_nu  

    # Convert to MJy/sr
    I_nu_MJy = I_nu * 1e20 * u.MJy / u.sr  

    # Convert to μK_CMB
    I_nu_μK = I_nu_MJy.to(u.K, equivalencies=u.brightness_temperature(nu * u.Hz)) * 1e6

    return I_nu_μK.value  # Return as a NumPy array

def MBB_factor(T_d, nu, nu0=353e9, sigma_nu0=7.1e-27, beta=1.53):
    """
    Computes the factor of the modified blackbody spectrum.

    Parameters:
    - T_d : float
        Constant dust temperature (in K).
    - nu : float
        Observed frequency (in Hz).
    - nu0 : float, optional
        Reference frequency (default: 353 GHz).
    - sigma_nu0 : float, optional
        Mean dust opacity at reference frequency (default: 7.1e-27 cm^2 per H at 353 GHz, Planck XVII).
    - beta : float, optional
        Dust emissivity spectral index (default: 1.53, Planck XVII).

    Returns:
    - I_nu_μK : 2D numpy array
        Observed intensity map converted to μK_CMB.
    """
    # Compute frequency-dependent dust opacity
    sigma_nu = sigma_nu0 * (nu / nu0) ** beta

    # Compute the Planck function using numerically stable expm1
    x = h * nu / (k * T_d)
    B_nu = (2 * h * nu**3 / c**2) / np.expm1(x)

    # Compute intensity map in W·m^-2·Hz^-1·sr^-1
    I_nu = sigma_nu * B_nu  

    # Convert to MJy/sr
    I_nu_MJy = I_nu * 1e20 * u.MJy / u.sr  

    # Convert to μK_CMB
    I_nu_μK = I_nu_MJy.to(u.K, equivalencies=u.brightness_temperature(nu * u.Hz)) * 1e6

    return I_nu_μK.value  # Return as a NumPy array

def downsample_by_four(image):
    """
    Downsamples a NumPy image by applying a Gaussian filter and averaging 
    over non-overlapping 2x2 pixel blocks.

    Args:
        image (np.ndarray): Input image of shape (H, W) for grayscale 
                            or (H, W, C) for color.

    Returns:
        np.ndarray: Downsampled image of shape (H/2, W/2) for grayscale 
                    or (H/2, W/2, C) for color.
    """
    H, W = image.shape[:2]

    # Ensure image dimensions are even
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError("Image dimensions must be even for 2x2 downsampling.")

    # Apply Gaussian filter with sigma=1 (approximately matches the 2x2 block size)
    # image = gaussian_filter(image, sigma=1)

    # Downsample by averaging over 2x2 blocks
    if image.ndim == 2:  # Grayscale
        return image.reshape(H//2, 2, W//2, 2).mean(axis=(1, 3))

    elif image.ndim == 3:  # Color (H, W, C)
        return image.reshape(H//2, 2, W//2, 2, image.shape[2]).mean(axis=(1, 3))

    else:
        raise ValueError("Unsupported image shape. Must be 2D (grayscale) or 3D (color).")