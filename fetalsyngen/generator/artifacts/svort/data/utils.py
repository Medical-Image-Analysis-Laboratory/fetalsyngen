from typing import Tuple, List, Optional, Collection, Sequence
import torch
import torch.nn.functional as F
from fetalsyngen.generator.artifacts.svort.definitions import DeviceType
from math import log, sqrt


GAUSSIAN_FWHM = 1 / (2 * sqrt(2 * log(2)))
SINC_FWHM = 1.206709128803223 * GAUSSIAN_FWHM


# # # # # # # # # # # # # # # # # #
#       Scanning utilities:
# Inteleaved sampling and PSF
# # # # # # # # # # # # # # # # # #


def interleave_index(N, n_i):
    idx = [None] * N
    t = 0
    for i in range(n_i):
        j = i
        while j < N:
            idx[j] = t
            t += 1
            j += n_i
    return idx


def resolution2sigma(rx, ry=None, rz=None, /, isotropic=False):
    """Define the spread of the PSF in terms of standard deviation, from the resolution.
    From the code of NeSVoR.
    """
    if isotropic:
        fx = fy = fz = GAUSSIAN_FWHM
    else:
        fx = fy = SINC_FWHM
        fz = GAUSSIAN_FWHM
    assert not ((ry is None) ^ (rz is None))
    if ry is None:
        if isinstance(rx, float) or isinstance(rx, int):
            if isotropic:
                return fx * rx
            else:
                return fx * rx, fy * rx, fz * rx
        elif isinstance(rx, torch.Tensor):
            if isotropic:
                return fx * rx
            else:
                assert rx.shape[-1] == 3
                return rx * torch.tensor([fx, fy, fz], dtype=rx.dtype, device=rx.device)
        elif isinstance(rx, List) or isinstance(rx, Tuple):
            assert len(rx) == 3
            return resolution2sigma(rx[0], rx[1], rx[2], isotropic=isotropic)
        else:
            raise Exception(str(type(rx)))
    else:
        return fx * rx, fy * ry, fz * rz


def get_PSF(
    r_max: Optional[int] = None,
    res_ratio: Tuple[float, float, float] = (1, 1, 3),
    threshold: float = 1e-4,
    device: DeviceType = torch.device("cpu"),
    psf_type: str = "gaussian",
) -> torch.Tensor:
    """
    Get the point spread function. This should do the same as the original get_PSF functions.
    From the code of NeSVoR.
    """

    sigma_x, sigma_y, sigma_z = resolution2sigma(res_ratio, isotropic=False)

    if r_max is None:
        r_max = max(int(2 * r + 1) for r in (sigma_x, sigma_y, sigma_z))
        r_max = max(r_max, 4)

    x = torch.linspace(-r_max, r_max, 2 * r_max + 1, dtype=torch.float32, device=device)
    grid_z, grid_y, grid_x = torch.meshgrid(x, x, x, indexing="ij")
    if psf_type == "gaussian":
        psf = torch.exp(
            -0.5 * (grid_x**2 / sigma_x**2 + grid_y**2 / sigma_y**2 + grid_z**2 / sigma_z**2)
        )
    elif psf_type == "sinc":
        psf = torch.sinc(
            torch.sqrt((grid_x / res_ratio[0]) ** 2 + (grid_y / res_ratio[1]) ** 2)
        ) ** 2 * torch.exp(-0.5 * grid_z**2 / sigma_z**2)
    else:
        raise TypeError(f"Unknown PSF type: <{psf_type}>!")
    psf[psf.abs() < threshold] = 0

    rx = int(torch.nonzero(psf.sum((0, 1)) > 0)[0, 0].item())
    ry = int(torch.nonzero(psf.sum((0, 2)) > 0)[0, 0].item())
    rz = int(torch.nonzero(psf.sum((1, 2)) > 0)[0, 0].item())
    psf = psf[
        rz : 2 * r_max + 1 - rz,
        ry : 2 * r_max + 1 - ry,
        rx : 2 * r_max + 1 - rx,
    ].contiguous()
    psf = psf / psf.sum()
    return psf


# # # # # # # # # # # # # # # # # #
# Utilities
# # # # # # # # # # # # # # # # # #


def resample(x: torch.Tensor, res_xyz_old: Sequence, res_xyz_new: Sequence) -> torch.Tensor:
    """
    Resample a tensor from a grid with resolution res_xyz_old to a grid with resolution res_xyz_new using
    pytorch's grid_sample function.
    """
    ndim = x.ndim - 2
    assert len(res_xyz_new) == len(res_xyz_old) == ndim
    if all(r_new == r_old for (r_new, r_old) in zip(res_xyz_new, res_xyz_old)):
        return x
    grids = []
    for i in range(ndim):
        fac = res_xyz_old[i] / res_xyz_new[i]
        size_new = int(x.shape[-i - 1] * fac)
        grid_max = (size_new - 1) / fac / (x.shape[-i - 1] - 1)
        grids.append(torch.linspace(-grid_max, grid_max, size_new, dtype=x.dtype, device=x.device))
    grid = torch.stack(torch.meshgrid(*grids[::-1], indexing="ij")[::-1], -1)
    y = F.grid_sample(
        x,
        grid[None].expand((x.shape[0],) + (-1,) * (ndim + 1)),
        align_corners=True,
    )
    # Interpolation done using bilinear interpolation by default, other options are
    # mode = 'nearest' | 'bicubic'
    # From pytorch documentation.
    # mode (str) – interpolation mode to calculate output values 'bilinear' | 'nearest' | 'bicubic'. Default: 'bilinear' Note: mode='bicubic' supports only 4-D input. When mode='bilinear' and the input is 5-D, the interpolation mode used internally will actually be trilinear. However, when the input is 4-D, the interpolation mode will legitimately be bilinear.
    return y


def meshgrid(
    shape_xyz: Collection,
    resolution_xyz: Collection,
    min_xyz: Optional[Collection] = None,
    device: DeviceType = None,
    stack_output: bool = True,
):
    assert len(shape_xyz) == len(resolution_xyz)
    if min_xyz is None:
        min_xyz = tuple(-(s - 1) * r / 2 for s, r in zip(shape_xyz, resolution_xyz))
    else:
        assert len(shape_xyz) == len(min_xyz)

    if device is None:
        if isinstance(shape_xyz, torch.Tensor):
            device = shape_xyz.device
        elif isinstance(resolution_xyz, torch.Tensor):
            device = resolution_xyz.device
        else:
            device = torch.device("cpu")
    dtype = torch.float32

    arr_xyz = [
        torch.arange(s, dtype=dtype, device=device) * r + m
        for s, r, m in zip(shape_xyz, resolution_xyz, min_xyz)
    ]
    grid_xyz = torch.meshgrid(arr_xyz[::-1], indexing="ij")[::-1]
    if stack_output:
        return torch.stack(grid_xyz, -1)
    else:
        return grid_xyz
