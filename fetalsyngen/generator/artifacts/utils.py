import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import conv3d
from dataclasses import dataclass, asdict


@dataclass
class ScannerParams:
    """
    Parameters that can vary between the mild and severe
    generator configurations.
    """

    resolution_slice_fac_min: float
    resolution_slice_fac_max: float
    resolution_slice_max: int
    slice_thickness_min: float
    slice_thickness_max: float
    gap_min: float
    gap_max: float
    min_num_stack: int
    max_num_stack: int
    max_num_slices: int
    noise_sigma_min: float
    noise_sigma_max: float
    TR_min: float
    TR_max: float
    prob_void: float
    prob_gamma: float
    gamma_std: float
    slice_size: int
    restrict_transform: bool
    txy: float
    resolution_recon: float = None
    slice_noise_threshold: float = 0.1


@dataclass
class ReconParams:
    prob_misreg_slice: float
    slices_misreg_ratio: float
    prob_misreg_stack: float
    txy: float
    prob_smooth: float
    prob_rm_slices: float
    rm_slices_min: float
    rm_slices_max: float
    prob_merge: float
    merge_ngaussians_min: int
    merge_ngaussians_max: int


def make_gaussian_kernel(sigma, device):
    """Taken from https://github.com/peirong26/Brain-ID/blob/main/BrainID/datasets/utils.py
    Utils needed for the synthetic data generation
    """
    sl = int(np.ceil(3 * sigma))
    ts = torch.linspace(-sl, sl, 2 * sl + 1, dtype=torch.float, device=device)
    gauss = torch.exp((-((ts / sigma) ** 2) / 2))
    kernel = gauss / gauss.sum()

    return kernel


def gaussian_blur_3d(input, stds, device):
    """Taken from https://github.com/peirong26/Brain-ID/blob/main/BrainID/datasets/utils.py
    Utils needed for the synthetic data generation
    """
    blurred = input[None, None, :, :, :]
    if stds[0] > 0:
        kx = make_gaussian_kernel(stds[0], device=device)
        blurred = conv3d(
            blurred,
            kx[None, None, :, None, None],
            stride=1,
            padding=(len(kx) // 2, 0, 0),
        )
    if stds[1] > 0:
        ky = make_gaussian_kernel(stds[1], device=device)
        blurred = conv3d(
            blurred,
            ky[None, None, None, :, None],
            stride=1,
            padding=(0, len(ky) // 2, 0),
        )
    if stds[2] > 0:
        kz = make_gaussian_kernel(stds[2], device=device)
        blurred = conv3d(
            blurred,
            kz[None, None, None, None, :],
            stride=1,
            padding=(0, 0, len(kz) // 2),
        )
    return torch.squeeze(blurred)


def mog_3d_tensor(shape, centers, sigmas, device):
    """
    Creates a 3D tensor with values following a Gaussian distribution centered at a specified point.

    Parameters:
    - shape: Tuple of three integers representing the dimensions of the tensor (D, H, W).
    - center: Tuple of three floats representing the center (x, y, z) of the Gaussian distribution.
    - sigma: Standard deviation of the Gaussian distribution.

    Returns:
    - A 3D tensor with values drawn from the Gaussian distribution centered at `center`.
    """
    D, H, W = shape
    # Create a 3D grid of coordinates
    x = torch.arange(W).float()
    y = torch.arange(H).float()
    z = torch.arange(D).float()
    x, y, z = torch.meshgrid(x, y, z, indexing="ij")
    x, y, z = x.to(device), y.to(device), z.to(device)
    mog = torch.zeros((D, H, W)).to(device)
    if not (isinstance(sigmas, list) or isinstance(sigmas, np.ndarray)):
        sigmas = [sigmas] * len(centers)

    for center, sigma in zip(centers, sigmas):
        if not isinstance(sigma, list) and not isinstance(sigma, np.ndarray):
            sigma_x, sigma_y, sigma_z = sigma, sigma, sigma
        else:
            sigma_x, sigma_y, sigma_z = sigma[0], sigma[1], sigma[2]
        x0, y0, z0 = center
        # Calculate the Gaussian distribution values
        dist_sq = (
            ((x - x0) / sigma_x) ** 2
            + ((y - y0) / sigma_y) ** 2
            + ((z - z0) / sigma_z) ** 2
        )
        mog += torch.exp(-dist_sq / 2)

    return torch.clamp(mog, 0, 1)


def apply_kernel(im, kernel_size=3):
    device = im.device
    kernel = torch.ones(
        (1, 1, kernel_size, kernel_size, kernel_size),
        dtype=torch.float32,
        device=device,
    )
    im = im.view(1, 1, *im.shape[-3:]).float()
    return F.conv3d(im.float(), kernel, padding=kernel_size // 2)


def erode(mask, kernel_size=3):
    """
    Perform erosion on a 3D binary mask using a 3D convolution.

    Args:
        mask (torch.Tensor): The input 3D binary mask (shape: [D, H, W]).
        kernel_size (int): The size of the kernel (default is 3).

    Returns:
        torch.Tensor: The eroded 3D binary mask.
    """
    # Perform 3D convolution
    eroded = apply_kernel(mask, kernel_size)

    # Threshold the result to obtain the eroded mask
    eroded = (eroded == kernel_size**3).int()

    return eroded.squeeze(0).squeeze(0)


def dilate(mask, kernel_size=3):
    """
    Perform dilation on a 3D binary mask using a 3D convolution.

    Args:
        mask (torch.Tensor): The input 3D binary mask (shape: [D, H, W]).
        kernel_size (int): The size of the kernel (default is 3).

    Returns:
        torch.Tensor: The dilated 3D binary mask.
    """
    # Perform 3D convolution
    dilated = apply_kernel(mask, kernel_size)

    # Threshold the result to obtain the dilated mask
    dilated = (dilated > 0).int()
    return dilated.squeeze(0).squeeze(0)
