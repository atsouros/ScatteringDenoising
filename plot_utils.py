import numpy as np
import torch
import matplotlib.pyplot as plt
import denoising
import utils
import importlib
import glob
utils = importlib.reload(utils)
plt.rcParams['text.usetex'] = True
from matplotlib.cm import ScalarMappable
from scipy.signal import windows

def margin_crop(image, identity = False):
    """
    Crops the image to a central square region of shape (target_size, target_size).

    Args:
        image (np.ndarray or torch.Tensor): Input image of shape (H, W) or (1, H, W).
        target_size (int): Desired size of the central square region to keep.

    Returns:
        Cropped image of shape (target_size, target_size) or (1, target_size, target_size).
    """

    H, W = image.shape[-2:]

    target_size = 256


    assert H == W, "Image must be square"
    assert target_size <= H, "Target size must be smaller than or equal to image size"

    margin = (H - target_size) // 2
    cropped = image[..., margin:margin + target_size, margin:margin + target_size]
    if not identity:
        return cropped
    else:
        return image
    
def low_pass_filter(image, k_max):
    if k_max is not None:
        H, W = image.shape
        kx = np.fft.fftfreq(W) * W
        ky = np.fft.fftfreq(H) * H
        kx, ky = np.meshgrid(kx, ky)
        k_squared = kx**2 + ky**2

        image_fft = np.fft.fft2(image)
        
        mask = k_squared <= k_max**2
        image_fft_filtered = image_fft * mask
        
        return np.fft.ifft2(image_fft_filtered).real
    else:
        return image
    
def plot_PS(stokes, image_Q, image_U, data_Q, data_U, nuisance, k_max):

    def apply_window(field):
        """Apply a 2D Blackman-Harris window to suppress edge effects."""
        ny, nx = field.shape
        win_y = windows.blackmanharris(ny)
        win_x = windows.blackmanharris(nx)
        window_2d = np.outer(win_y, win_x)
        return field * window_2d

    plt.figure(figsize=(8, 6))

    if stokes == 'Q':
        image = image_Q
        data = data_Q
        nuisances = nuisance[:, 0]
    else:
        image = image_U
        data = data_U
        nuisances = nuisance[:, 1]

    nx, ny = data.shape
    lam = 0.55
    image = denoising.filter_radial(np.array([image]), lambda k: k < nx * lam)[0]

    # Apply window
    data_win = apply_window(data)
    image_win = apply_window(image)
    nuisances_win = np.array([apply_window(c) for c in nuisances])

    # Compute power spectra
    k_data, P_data = utils.power_spectrum(data_win)
    k_image, P_image = utils.power_spectrum(image_win)
    k_noise, P_noise = utils.power_spectrum(data_win - image_win)

    k_all_c, P_c_all = zip(*[utils.power_spectrum(c) for c in nuisances_win])
    k_ref = np.array(k_all_c[0])
    P_c_all = np.array(P_c_all)
    P_c_mean = np.mean(P_c_all, axis=0)
    P_c_std = np.std(P_c_all, axis=0)

    k_all_sc, P_sc_all = zip(*[utils.power_spectrum(image_win + c) for c in nuisances_win])
    P_sc_all = np.array(P_sc_all)
    P_sc_mean = np.mean(P_sc_all, axis=0)
    P_sc_std = np.std(P_sc_all, axis=0)

    def trim(k, P):
        if k_max is not None:
            mask = k_ref < k_max
            return k[mask], P[mask]
        else:
            return k, P

    k_data, P_data = trim(k_data, P_data)
    k_image, P_image = trim(k_image, P_image)
    k_noise, P_noise = trim(k_noise, P_noise)
    k_mean, P_sc_mean = trim(k_ref, P_sc_mean)
    _, P_sc_std = trim(k_ref, P_sc_std)
    _, P_c_mean = trim(k_ref, P_c_mean)
    _, P_c_std = trim(k_ref, P_c_std)

    # Plot
    plt.loglog(k_data, P_data, label=r"$d$", c="r")
    plt.loglog(k_image, P_image, label=r"$\tilde{s}$", c="b")
    plt.loglog(k_noise, P_noise, label=r"$d - \tilde{s}$", c="orange")
    plt.loglog(k_mean, P_sc_mean, label=r"$\langle \tilde{s} + c \rangle$", c="c")
    plt.fill_between(
        k_mean,
        P_sc_mean - P_sc_std,
        P_sc_mean + P_sc_std,
        color="c",
        alpha=0.3,
        linewidth=0,
        label=r"$\langle \tilde{s} + c \rangle \pm \sigma$"
    )
    plt.loglog(k_mean, P_c_mean, label=r"$\langle c \rangle$", c="g")
    plt.fill_between(
        k_mean,
        P_c_mean - P_c_std,
        P_c_mean + P_c_std,
        color="g",
        alpha=0.3,
        linewidth=0,
        label=r"$\langle c \rangle \pm \sigma$"
    )

    plt.xlabel(r"$k$", fontsize=16)
    plt.ylabel(r"$P(k)$", fontsize=16)
    plt.title(fr"Stokes-${stokes}$")
    plt.legend()
    plt.show()

def plot_CS(stokes, image_Q, image_U, data_Q, data_U, nuisance, k_max=20):
    plt.figure(figsize=(8, 6))

    def apply_window(field):
        ny, nx = field.shape
        win_y = windows.blackmanharris(ny)
        win_x = windows.blackmanharris(nx)
        return field * np.outer(win_y, win_x)

    if stokes == 'Q':
        image = image_Q
        data = data_Q
        nuisances = nuisance[:, 0]
    else:
        image = image_U
        data = data_U
        nuisances = nuisance[:, 1]

    nx, ny = data.shape
    lam = 0.55
    image_filtered = denoising.filter_radial(np.array([image]), lambda k: k < nx * lam)[0]

    # Apply windowing
    image_win = apply_window(image_filtered)
    data_win = apply_window(data)
    nuisances_win = np.array([apply_window(c) for c in nuisances])

    # Residual and its cross-spectrum
    residual_win = data_win - image_win
    k_res, P_cross_residual = utils.cross_spectrum(image_win, residual_win)

    # Cross-spectrum with each nuisance realization
    k_all, cs_all = zip(*[utils.cross_spectrum(image_win, c) for c in nuisances_win])
    k_mean = np.array(k_all[0])
    cs_all = np.array(cs_all)
    cs_mean = np.mean(cs_all, axis=0)
    cs_std = np.std(cs_all, axis=0)

    # Trim helper
    def trim(k, P):
        if k_max is not None:
            mask = k < k_max
            return k[mask], P[mask]
        else:
            return k, P

    # Trim all
    k_res, P_cross_residual = trim(k_res, np.abs(P_cross_residual))
    k_mean_trim, cs_mean = trim(k_mean, np.abs(cs_mean))
    _, cs_std = trim(k_mean, cs_std)

    # Plot
    plt.loglog(k_res, P_cross_residual, label=r"$|\tilde{s} \times (d - \tilde{s})|$", c="m")
    plt.loglog(k_mean_trim, cs_mean, label=r"$|\langle \tilde{s} \times c \rangle|$", c="orange")
    plt.fill_between(
        k_mean_trim,
        cs_mean - cs_std,
        cs_mean + cs_std,
        color="orange",
        alpha=0.3,
        linewidth=0,
        label=r"$\langle \tilde{s} \times c \rangle \pm \sigma$"
    )

    plt.xlabel(r"$k$", fontsize=16)
    plt.ylabel(r"$P_{\times}(k)$", fontsize=16)
    plt.title(fr"Cross-spectra: Stokes-${stokes}$", fontsize=16)
    plt.legend()
    plt.show()

def plot_maps(stokes, image_Q, image_U, data_Q, data_U, nuisance, ran, cmap = 'plasma', vmin = None, vmax = None):
    def apply_window(field):
        """Apply a 2D Blackman-Harris window to suppress edge effects."""
        ny, nx = field.shape
        win_y = windows.blackmanharris(ny)
        win_x = windows.blackmanharris(nx)
        window_2d = np.outer(win_y, win_x)
        return field * window_2d

    # Select appropriate Stokes fields
    if stokes == 'Q':
        image = image_Q
        data = data_Q
        c_i = nuisance[ran, 0]
    else:
        image = image_U
        data = data_U
        c_i = nuisance[ran, 1]

    combined = data + c_i

    # Fixed colorbar scale
    color = 'white'

    # Set up figure with single row
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.patch.set_alpha(0)
    plt.rcParams['text.usetex'] = True

    titles = [r"$d$", r"$\tilde{s}$", r"$\tilde{s} + c_i$"]
    maps = [data, image, combined]

    for ax, title, m in zip(axes, titles, maps):
        if torch.is_tensor(m):
            m = m.cpu().numpy()
        ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=16, color=color)
        ax.axis('off')

    # Shared colorbar
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label(r"$MJy/sr$", fontsize=20, color=color)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=color)

    fig.suptitle(fr"Stokes-${stokes}$", fontsize=20, color='white')
    plt.show()

    
def plot_maps_GNILC(stokes, image_Q, image_U, data_Q, data_U, Q_GNILC, U_GNILC, ran,  vmin = None, vmax = None, cmap = 'plasma'):

    def apply_window(field):
        """Apply a 2D Blackman-Harris window to suppress edge effects."""
        ny, nx = field.shape
        win_y = windows.blackmanharris(ny)
        win_x = windows.blackmanharris(nx)
        window_2d = np.outer(win_y, win_x)
        return field * window_2d
    

    # Select appropriate Stokes fields
    if stokes == 'Q':
        image = image_Q
        data = data_Q
        gnilc = Q_GNILC
        c_i = nuisance[np.random.randint(ran), 0]
    else:
        image = image_U
        data = data_U
        gnilc = U_GNILC
        c_i = nuisance[np.random.randint(ran), 1]

    # image = apply_window(image)
    # data = apply_window(data)
    # gnilc = apply_window(gnilc)
    # c_i = apply_window(c_i)

    # image = np.fft.fftshift(image)
    # data = np.fft.fftshift(data)
    # gnilc = np.fft.fftshift(gnilc)
    # c_i = np.fft.fftshift(c_i)

    # Compute combined map
    combined = data + c_i

    # Fixed colorbar scale
    vmin, vmax = -0.1, 0.15
    color = 'white'

    # Set up figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    fig.patch.set_alpha(0)

    # Enable LaTeX
    plt.rcParams['text.usetex'] = True

    # Plot maps
    titles = [r"$d$", r"$\tilde{s}$", r"$\tilde{s} + c_i$", r"GNILC"]
    maps = [data, image, combined, gnilc]

    for ax, title, m in zip(axes.flat, titles, maps):
        # Convert to numpy if needed
        if torch.is_tensor(m):
            m = m.cpu().numpy()
        ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=16, color=color)
        ax.axis('off')

    # Colorbar from fixed scalar mappable
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label(r"$MJy/sr$", fontsize=20, color=color)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=color)

    # Add figure title
    fig.suptitle(fr"Stokes-${stokes}$", fontsize=20, color='white')

    plt.show()

def plot_noise(stokes, image_Q, image_U, data_Q, data_U, nuisance, ran,  vmin = None, vmax = None, cmap = 'plasma'):
    # Select appropriate Stokes fields
    if stokes == 'Q':
        image = image_Q
        data = data_Q
        c_i = nuisance[ran, 0]
    else:
        image = image_U
        data = data_U
        c_i = nuisance[ran, 1]

    # Compute residual
    residual = data - image

    # residual = np.fft.fftshift(residual)
    # c_i = np.fft.fftshift(c_i)

    # Common settings
    color = 'white'

    # Determine global color limits
    # vmin = min(residual.min(), c_i.min())
    # vmax = max(residual.max(), c_i.max())

    vmin, vmax = -0.1, 0.15


    # Set up figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.patch.set_alpha(0)

    # Enable LaTeX
    plt.rcParams['text.usetex'] = True

    # Plot residual
    axes[0].imshow(residual, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(r"$d - \tilde{s}$", fontsize=16, color=color)
    axes[0].axis('off')

    # Plot c_i
    im1 = axes[1].imshow(c_i, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(r"$c_i$", fontsize=16, color=color)
    axes[1].axis('off')

    # Add colorbar
    cbar = fig.colorbar(im1, ax=axes, orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label(r"$MJy/sr$", fontsize=20, color=color)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=color)

    # Add figure title
    # fig.suptitle(fr"Stokes-${stokes}$", fontsize=20, color='white')

    plt.show()