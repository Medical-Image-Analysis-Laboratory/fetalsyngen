from __future__ import annotations

import torch
import numpy as np
from .transform_convert import (
    axisangle2mat,
    mat2axisangle,
)
from scipy.spatial.transform import Rotation
from fetalsynthgen.generation.svort.definitions import DeviceType
from typing import Iterable, Union, Tuple


class RigidTransform(object):
    def __init__(self, data, trans_first=True, device=None):
        self.trans_first = trans_first
        self._axisangle = None
        self._matrix = None
        if device is not None:
            data = data.to(device)
        if data.shape[1] == 6:  # parameter
            self._axisangle = data
            self._matrix = None
        elif data.shape[1] == 3:  # matrix
            self._axisangle = None
            self._matrix = data
        else:
            raise Exception("Unknown format for rigid transform!")

    def matrix(self, trans_first=True):
        if self._matrix is not None:
            mat = self._matrix
        else:
            mat = axisangle2mat(self._axisangle)
        if self.trans_first == True and trans_first == False:
            mat = mat_first2last(mat)
        elif self.trans_first == False and trans_first == True:
            mat = mat_last2first(mat)
        return mat

    def axisangle(self, trans_first=True):
        if self._axisangle is not None:
            ax = self._axisangle
        else:
            ax = mat2axisangle(self._matrix)

        if self.trans_first == True and trans_first == False:
            ax = ax_first2last(ax)
        elif self.trans_first == False and trans_first == True:
            ax = ax_last2first(ax)
        return ax

    def inv(self):
        mat = self.matrix(trans_first=True)
        R = mat[:, :, :3]
        t = mat[:, :, 3:]
        mat = torch.cat((R.transpose(-2, -1), -torch.matmul(R, t)), -1)
        return RigidTransform(mat, trans_first=True)

    def compose(self, other):
        mat1 = self.matrix(trans_first=True)
        mat2 = other.matrix(trans_first=True)
        R1 = mat1[:, :, :3]
        t1 = mat1[:, :, 3:]
        R2 = mat2[:, :, :3]
        t2 = mat2[:, :, 3:]
        R = torch.matmul(R1, R2)
        t = t2 + torch.matmul(R2.transpose(-2, -1), t1)
        mat = torch.cat((R, t), -1)
        return RigidTransform(mat, trans_first=True)

    def __getitem__(self, idx):
        if self._axisangle is not None:
            data = self._axisangle[idx]
            if len(data.shape) < 2:
                data = data.unsqueeze(0)
        else:
            data = self._matrix[idx]
            if len(data.shape) < 3:
                data = data.unsqueeze(0)
        return RigidTransform(data, self.trans_first)

    def detach(self):
        if self._axisangle is not None:
            data = self._axisangle.detach()
        else:
            data = self._matrix.detach()
        return RigidTransform(data, self.trans_first)

    @property
    def device(self) -> DeviceType:
        if self._axisangle is not None:
            return self._axisangle.device
        elif self._matrix is not None:
            return self._matrix.device
        else:
            raise Exception("Both data are None!")

    def dtype(self) -> torch.dtype:
        if self._axisangle is not None:
            return self._axisangle.dtype
        elif self._matrix is not None:
            return self._matrix.dtype
        else:
            raise Exception("Both data are None!")

    @staticmethod
    def cat(transforms: Iterable[RigidTransform]) -> RigidTransform:
        matrixs = [t.matrix(trans_first=True) for t in transforms]
        return RigidTransform(torch.cat(matrixs, 0), trans_first=True)

    def __len__(self) -> int:
        if self._axisangle is not None:
            return self._axisangle.shape[0]
        elif self._matrix is not None:
            return self._matrix.shape[0]
        else:
            raise Exception("Both data are None!")

    def mean(self, trans_first=True, simple_mean=True) -> RigidTransform:
        ax = self.axisangle(trans_first=trans_first)
        if simple_mean:
            ax_mean = ax.mean(0, keepdim=True)
        else:
            meanT = ax[:, 3:].mean(0, keepdim=True)
            meanR = average_rotation(ax[:, :3])
            ax_mean = torch.cat((meanR, meanT), -1)
        return RigidTransform(ax_mean, trans_first=trans_first)


"""helper for RigidTransform"""


def mat_first2last(mat):
    R = mat[:, :, :3]
    t = mat[:, :, 3:]
    t = torch.matmul(R, t)
    mat = torch.cat([R, t], -1)
    return mat


def mat_last2first(mat):
    R = mat[:, :, :3]
    t = mat[:, :, 3:]
    t = torch.matmul(R.transpose(-2, -1), t)
    mat = torch.cat([R, t], -1)
    return mat


def ax_first2last(axisangle):
    mat = axisangle2mat(axisangle)
    mat = mat_first2last(mat)
    return mat2axisangle(mat)


def ax_last2first(axisangle):
    mat = axisangle2mat(axisangle)
    mat = mat_last2first(mat)
    return mat2axisangle(mat)


def mat_update_resolution(mat, res_from, res_to):
    assert mat.dim() == 3
    fac = torch.ones_like(mat[:1, :1])

    fac[..., 3] = res_from / res_to
    return mat * fac


def ax_update_resolution(ax, res_from, res_to):
    assert ax.dim() == 2
    fac = torch.ones_like(ax[:1])
    fac[:, 3:] = res_from / res_to
    return ax * fac


# random angle
def random_angle(n, restricted, device):
    a = 2 * np.pi * np.random.rand(n)
    b = np.arccos(2 * np.random.rand(n) - 1)
    if restricted:
        c = np.pi * np.random.rand(n)
    else:
        c = np.pi * (2 * np.random.rand(n) - 1)

    R = Rotation.from_euler("ZXZ", np.stack([a, b, c], -1))
    R = R.as_rotvec()
    return torch.from_numpy(R).to(dtype=torch.float32, device=device)


def random_trans(n, T_range, device):
    if not isinstance(T_range, (list, tuple)):
        T_range = [T_range, T_range, T_range]
    else:
        assert len(T_range) == 3
    tx = (torch.rand(n, device=device) - 0.5) * T_range[0]
    ty = (torch.rand(n, device=device) - 0.5) * T_range[1]
    tz = (torch.rand(n, device=device) - 0.5) * T_range[2]
    return torch.stack([tx, ty, tz], -1)


"""misc"""


def mat2euler(mat):
    TOL = 0.000001
    TX = mat[:, 0, 3]
    TY = mat[:, 1, 3]
    TZ = mat[:, 2, 3]

    tmp = torch.asin(-mat[:, 0, 2])
    mask = torch.cos(tmp).abs() <= TOL
    RX = torch.atan2(mat[:, 1, 2], mat[:, 2, 2])
    RY = tmp
    RZ = torch.atan2(mat[:, 0, 1], mat[:, 0, 0])
    RX[mask] = torch.atan2(
        -mat[:, 0, 2] * mat[:, 1, 0], -mat[:, 0, 2] * mat[:, 2, 0]
    )[mask]
    RZ[mask] = 0

    RX *= 180 / np.pi
    RY *= 180 / np.pi
    RZ *= 180 / np.pi

    return torch.stack((TX, TY, TZ, RX, RY, RZ), -1)


def euler2mat(p):
    tx = p[:, 0]
    ty = p[:, 1]
    tz = p[:, 2]

    rx = p[:, 3]
    ry = p[:, 4]
    rz = p[:, 5]

    M_PI = np.pi
    cosrx = torch.cos(rx * (M_PI / 180.0))
    cosry = torch.cos(ry * (M_PI / 180.0))
    cosrz = torch.cos(rz * (M_PI / 180.0))
    sinrx = torch.sin(rx * (M_PI / 180.0))
    sinry = torch.sin(ry * (M_PI / 180.0))
    sinrz = torch.sin(rz * (M_PI / 180.0))

    mat = torch.eye(4, device=p.device)
    mat = mat.reshape((1, 4, 4)).repeat(p.shape[0], 1, 1)

    mat[:, 0, 0] = cosry * cosrz
    mat[:, 0, 1] = cosry * sinrz
    mat[:, 0, 2] = -sinry
    mat[:, 0, 3] = tx

    mat[:, 1, 0] = sinrx * sinry * cosrz - cosrx * sinrz
    mat[:, 1, 1] = sinrx * sinry * sinrz + cosrx * cosrz
    mat[:, 1, 2] = sinrx * cosry
    mat[:, 1, 3] = ty

    mat[:, 2, 0] = cosrx * sinry * cosrz + sinrx * sinrz
    mat[:, 2, 1] = cosrx * sinry * sinrz - sinrx * cosrz
    mat[:, 2, 2] = cosrx * cosry
    mat[:, 2, 3] = tz
    mat[:, 3, 3] = 1.0

    return mat


def point2mat(p):
    p = p.view(-1, 3, 3)
    p1 = p[:, 0]
    p2 = p[:, 1]
    p3 = p[:, 2]
    v1 = p3 - p1
    v2 = p2 - p1

    nz = torch.cross(v1, v2, -1)
    ny = torch.cross(nz, v1, -1)
    nx = v1

    R = torch.stack((nx, ny, nz), -1)
    R = R / torch.linalg.norm(R, ord=2, dim=-2, keepdim=True)

    T = torch.matmul(R.transpose(-2, -1), p2.unsqueeze(-1))

    return torch.cat((R, T), -1)


def mat2point(mat, sx, sy, rs):
    p1 = torch.tensor([-(sx - 1) / 2 * rs, -(sy - 1) / 2 * rs, 0]).to(
        dtype=mat.dtype, device=mat.device
    )
    p2 = torch.tensor([0, 0, 0]).to(dtype=mat.dtype, device=mat.device)
    p3 = torch.tensor([(sx - 1) / 2 * rs, -(sy - 1) / 2 * rs, 0]).to(
        dtype=mat.dtype, device=mat.device
    )
    p = torch.stack((p1, p2, p3), 0)
    p = p.unsqueeze(0).unsqueeze(-1)  # 1x3x3x1
    R = mat[:, :, :-1].unsqueeze(1)  # nx1x3x3
    T = mat[:, :, -1:].unsqueeze(1)  # nx1x3x1
    p = torch.matmul(R, p + T)
    return p.view(-1, 9)


def average_rotation(R: torch.Tensor) -> torch.Tensor:
    import scipy
    from scipy.spatial.transform import Rotation

    dtype = R.dtype
    device = R.device
    Rmat = Rotation.from_rotvec(R.cpu().numpy()).as_matrix()
    R = Rotation.from_rotvec(R.cpu().numpy()).as_quat()
    for i in range(R.shape[0]):
        if np.linalg.norm(R[i] + R[0]) < np.linalg.norm(R[i] - R[0]):
            R[i] *= -1
    barR = np.mean(R, 0)
    barR = barR / np.linalg.norm(barR)

    S_new = S = Rotation.from_quat(barR).as_matrix()
    R = Rmat
    i = 0
    while np.all(np.isreal(S_new)) and np.all(np.isfinite(S_new)) and i < 10:
        S = S_new
        i += 1
        sum_vmatrix_normed = np.zeros((3, 3))
        sum_inv_norm_vmatrix = 0
        for j in range(R.shape[0]):
            vmatrix = scipy.linalg.logm(np.matmul(R[j], np.linalg.inv(S)))
            vmatrix_normed = vmatrix / np.linalg.norm(
                vmatrix, ord=2, axis=(0, 1)
            )
            sum_vmatrix_normed += vmatrix_normed
            sum_inv_norm_vmatrix += 1 / np.linalg.norm(
                vmatrix, ord=2, axis=(0, 1)
            )

        delta = sum_vmatrix_normed / sum_inv_norm_vmatrix
        if np.all(np.isfinite(delta)):
            S_new = np.matmul(scipy.linalg.expm(delta), S)
        else:
            break

    S = Rotation.from_matrix(S).as_rotvec()
    return torch.tensor(S[None], dtype=dtype, device=device)


def get_transform_diff_mean(
    transform_out: RigidTransform,
    transform_in: RigidTransform,
    mean_r: int = 3,
) -> Tuple[RigidTransform, RigidTransform]:
    transform_diff = transform_out.compose(transform_in.inv())
    length = len(transform_diff)
    assert length > 0, "input is empty!"
    mid = length // 2
    left = max(0, mid - mean_r)
    right = min(length, mid + mean_r)
    transform_diff_mean = transform_diff[left:right].mean(simple_mean=False)
    return transform_diff_mean, transform_diff


# # # # # # # # # # # # # # # # # #
# Transformation initialization
# # # # # # # # # # # # # # # # # #


def random_init_stack_transforms(n_slice, gap, restricted, txy, device):
    """Initialize a stack of transforms. From the code of SVoRT."""
    angle = random_angle(1, restricted, device).expand(n_slice, -1)
    tz = (
        torch.arange(0, n_slice, device=device, dtype=torch.float32)
        - (n_slice - 1) / 2.0
    ) * gap
    if txy:
        tx = torch.ones_like(tz) * np.random.uniform(-txy, txy)
        ty = torch.ones_like(tz) * np.random.uniform(-txy, txy)
    else:
        tx = ty = torch.zeros_like(tz)
    t = torch.stack((tx, ty, tz), -1)
    return RigidTransform(torch.cat((angle, t), -1), trans_first=True)


def init_stack_transform(
    n_slice: int, gap: float, device: DeviceType
) -> RigidTransform:
    """Initialize a stack of transforms. From the code of NeSVoR."""
    ax = torch.zeros((n_slice, 6), dtype=torch.float32, device=device)
    ax[:, -1] = (
        torch.arange(n_slice, dtype=torch.float32, device=device)
        - (n_slice - 1) / 2.0
    ) * gap
    return RigidTransform(ax, trans_first=True)


def init_zero_transform(n: int, device: DeviceType) -> RigidTransform:
    """Initialize a stack of transforms. From the code of NeSVoR."""
    return RigidTransform(
        torch.zeros((n, 6), dtype=torch.float32, device=device)
    )


def reset_transform(transform):
    transform = transform.axisangle()
    transform[:, :-1] = 0
    transform[:, -1] -= transform[:, -1].mean()
    return RigidTransform(transform)


def mat_transform_points(
    mat: torch.Tensor, x: torch.Tensor, trans_first: bool
) -> torch.Tensor:
    """Coordinate-wise matrix transformation, from the code of NeSVoR."""
    # mat (*, 3, 4)
    # x (*, 3)
    R = mat[..., :-1]  # (*, 3, 3)
    T = mat[..., -1:]  # (*, 3, 1)
    x = x[..., None]  # (*, 3, 1)
    if trans_first:
        x = torch.matmul(R, x + T)  # (*, 3)
    else:
        x = torch.matmul(R, x) + T
    return x[..., 0]


def transform_points(
    transform: RigidTransform, x: torch.Tensor
) -> torch.Tensor:
    """Coordinate-wise transformation, from the code of NeSVoR."""
    # transform (N) and x (N, 3)
    # or transform (1) and x (*, 3)
    assert x.ndim == 2 and x.shape[-1] == 3
    trans_first = transform.trans_first
    mat = transform.matrix(trans_first)
    return mat_transform_points(mat, x, trans_first)


def compare_resolution_affine(r1, a1, r2, a2, s1, s2) -> bool:
    r1 = np.array(r1)
    a1 = np.array(a1)
    r2 = np.array(r2)
    a2 = np.array(a2)
    if s1 != s2:
        return False
    if r1.shape != r2.shape:
        return False
    if np.amax(np.abs(r1 - r2)) > 1e-3:
        return False
    if a1.shape != a2.shape:
        return False
    if np.amax(np.abs(a1 - a2)) > 1e-3:
        return False
    return True


def affine2transformation(
    volume: torch.Tensor,
    mask: torch.Tensor,
    resolutions: np.ndarray,
    affine: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, RigidTransform]:
    device = volume.device
    d, h, w = volume.shape

    R = affine[:3, :3]
    negative_det = np.linalg.det(R) < 0

    T = affine[:3, -1:]  # T = R @ (-T0 + T_r)
    R = R @ np.linalg.inv(np.diag(resolutions))

    T0 = np.array(
        [(w - 1) / 2 * resolutions[0], (h - 1) / 2 * resolutions[1], 0]
    )
    T = np.linalg.inv(R) @ T + T0.reshape(3, 1)

    tz = (
        torch.arange(0, d, device=device, dtype=torch.float32) * resolutions[2]
        + T[2].item()
    )
    tx = torch.ones_like(tz) * T[0].item()
    ty = torch.ones_like(tz) * T[1].item()
    t = torch.stack((tx, ty, tz), -1).view(-1, 3, 1)
    R = torch.tensor(R, device=device).unsqueeze(0).repeat(d, 1, 1)

    if negative_det:
        volume = torch.flip(volume, (-1,))
        mask = torch.flip(mask, (-1,))
        t[:, 0, -1] *= -1
        R[:, :, 0] *= -1

    transformation = RigidTransform(
        torch.cat((R, t), -1).to(torch.float32), trans_first=True
    )

    return volume, mask, transformation


def transformation2affine(
    volume: torch.Tensor,
    transformation: RigidTransform,
    resolution_x: float,
    resolution_y: float,
    resolution_z: float,
) -> np.ndarray:
    mat = transformation.matrix(trans_first=True).detach().cpu().numpy()
    assert mat.shape[0] == 1
    R = mat[0, :, :-1]
    T = mat[0, :, -1:]
    d, h, w = volume.shape
    affine = np.eye(4)
    T[0] -= (w - 1) / 2 * resolution_x
    T[1] -= (h - 1) / 2 * resolution_y
    T[2] -= (d - 1) / 2 * resolution_z
    T = R @ T.reshape(3, 1)
    R = R @ np.diag([resolution_x, resolution_y, resolution_z])
    affine[:3, :] = np.concatenate((R, T), -1)
    return affine
