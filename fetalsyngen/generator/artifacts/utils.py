import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import conv3d
from dataclasses import dataclass, field
import os
import time

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





class MergeParams:
    pass

@dataclass
class GaussianMergeParams(MergeParams):
    merge_type: str = field(init=False)
    ngaussians_min: int
    ngaussians_max: int
    def __post_init__(self):
        self.merge_type = "gaussian"

@dataclass
class PerlinMergeParams(MergeParams):
    merge_type: str = field(init=False)
    res_list: list[int]
    octaves_list: list[int]
    persistence: float
    lacunarity: float

    def __post_init__(self):
        self.merge_type = "perlin"

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
    merge_params: MergeParams

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

# ported from https://github.com/pvigier/perlin-numpy
# https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/perlin-noise-part-2/perlin-noise-computing-derivatives.html
# https://rtouti.github.io/graphics/perlin-noise-algorithm
# https://github.com/peirong26/UNA/blob/main/FluidAnomaly/perlin3d.py


def perlin_interpolant(t):
    # Perlin interpolation: 6t^5 - 15t^4 + 10t^3
    return t * t * t * (t * (t * 6 - 15) + 10)

def generate_perlin_noise_3d(shape, res, tileable=(False, False, False), interpolant=perlin_interpolant, device=None):
    """
    Generate a 3D torch tensor of Perlin noise.

    Args:
        shape: The shape of the generated array (tuple of three ints).
        res: The number of periods of noise to generate along each
            axis (tuple of three ints).
        tileable: If the noise should be tileable along each axis
            (tuple of three bools). Defaults to (False, False, False).
        interpolant: The interpolation function. Defaults to
            t*t*t*(t*(t*6 - 15) + 10).
        device: The device to use for the computation. Defaults to
            'cuda' if available, otherwise 'cpu'.

    Returns:
        A 3D torch tensor of Perlin noise with the specified shape.

    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    shape = torch.tensor(shape, device=device)
    res = torch.tensor(res, device=device)

    # Create 3D grid of coordinates
    lin = [torch.linspace(0, res[i], shape[i], device=device) for i in range(3)]
    grid = torch.stack(torch.meshgrid(*lin, indexing='ij'), dim=-1)  # shape (X,Y,Z,3)

    # Integer lattice coordinates (which cell)
    cell = grid.floor().to(torch.long)

    # Local coordinates inside each cell (0..1)
    local_xyz = grid - cell

    # Generate random gradient vectors at lattice points
    theta = 2 * torch.pi * torch.rand(res[0]+1, res[1]+1, res[2]+1, device=device)
    phi = 2 * torch.pi * torch.rand(res[0]+1, res[1]+1, res[2]+1, device=device)
    gradients = torch.stack((
        torch.sin(phi) * torch.cos(theta),
        torch.sin(phi) * torch.sin(theta),
        torch.cos(phi)
    ), dim=-1)  

    # Make gradients tileable if needed
    if tileable[0]:
        gradients[-1,:,:] = gradients[0,:,:]
    if tileable[1]:
        gradients[:,-1,:] = gradients[:,0,:]
    if tileable[2]:
        gradients[:,:,-1] = gradients[:,:,0]

    # Fetch gradient vectors at each corner of the cell
    def get_grad(ix, iy, iz):
        return gradients[
            ix.clamp(max=res[0]),
            iy.clamp(max=res[1]),
            iz.clamp(max=res[2])
        ]

    g000 = get_grad(cell[...,0],     cell[...,1],     cell[...,2])
    g100 = get_grad(cell[...,0] + 1, cell[...,1],     cell[...,2])
    g010 = get_grad(cell[...,0],     cell[...,1] + 1, cell[...,2])
    g110 = get_grad(cell[...,0] + 1, cell[...,1] + 1, cell[...,2])
    g001 = get_grad(cell[...,0],     cell[...,1],     cell[...,2] + 1)
    g101 = get_grad(cell[...,0] + 1, cell[...,1],     cell[...,2] + 1)
    g011 = get_grad(cell[...,0],     cell[...,1] + 1, cell[...,2] + 1)
    g111 = get_grad(cell[...,0] + 1, cell[...,1] + 1, cell[...,2] + 1)

    # Compute vectors from each corner to current point
    def dot_grid_gradient(grad, x_offset, y_offset, z_offset):
        offset = torch.tensor([x_offset, y_offset, z_offset], device=device)
        delta = local_xyz - offset
        return (grad * delta).sum(dim=-1)

    n000 = dot_grid_gradient(g000, 0.0, 0.0, 0.0)
    n100 = dot_grid_gradient(g100, 1.0, 0.0, 0.0)
    n010 = dot_grid_gradient(g010, 0.0, 1.0, 0.0)
    n110 = dot_grid_gradient(g110, 1.0, 1.0, 0.0)
    n001 = dot_grid_gradient(g001, 0.0, 0.0, 1.0)
    n101 = dot_grid_gradient(g101, 1.0, 0.0, 1.0)
    n011 = dot_grid_gradient(g011, 0.0, 1.0, 1.0)
    n111 = dot_grid_gradient(g111, 1.0, 1.0, 1.0)

    # Compute Perlin interpolation weights
    t = interpolant(local_xyz)

    # Interpolate
    n00 = n000 * (1 - t[...,0]) + t[...,0] * n100
    n10 = n010 * (1 - t[...,0]) + t[...,0] * n110
    n01 = n001 * (1 - t[...,0]) + t[...,0] * n101
    n11 = n011 * (1 - t[...,0]) + t[...,0] * n111

    n0 = n00 * (1 - t[...,1]) + t[...,1] * n10
    n1 = n01 * (1 - t[...,1]) + t[...,1] * n11

    noise = n0 * (1 - t[...,2]) + t[...,2] * n1

    return noise



def generate_fractal_noise_3d(
        shape, res, octaves=1, persistence=0.5, lacunarity=2,
        tileable=(False, False, False), interpolant=perlin_interpolant, device=None
):
    """Generate a 3D numpy array of fractal noise.

    Args:
        shape: The shape of the generated array (tuple of three ints).
            This must be a multiple of lacunarity**(octaves-1)*res.
        res: The number of periods of noise to generate along each
            axis (tuple of three ints). Note shape must be a multiple of
            (lacunarity**(octaves-1)*res).
        octaves: The number of octaves in the noise. Defaults to 1.
        persistence: The scaling factor between two octaves.
        lacunarity: The frequency factor between two octaves.
        tileable: If the noise should be tileable along each axis
            (tuple of three bools). Defaults to (False, False, False).
        interpolant: The, interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of fractal noise and of shape generated by
        combining several octaves of perlin noise.

    Raises:
        ValueError: If shape is not a multiple of
            (lacunarity**(octaves-1)*res).
    """
    seed = int(time.time())
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) 
    
    noise = torch.zeros(shape)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    noise = noise.to(device)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_3d(
            shape,
            (frequency*res[0], frequency*res[1], frequency*res[2]),
            tileable,
            interpolant,
            device,
        )
        frequency *= lacunarity
        amplitude *= persistence

   
    return noise
    