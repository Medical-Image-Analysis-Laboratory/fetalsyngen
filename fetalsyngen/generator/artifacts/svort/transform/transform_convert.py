from torch import nn
from torch.autograd import Function
import torch
from torch.utils.cpp_extension import load
import os

dirname = os.path.dirname(__file__)

transform_convert_cuda = load(
    "transform_convert_cuda",
    [
        os.path.join(dirname, "transform_convert_cuda.cpp"),
        os.path.join(dirname, "transform_convert_cuda_kernel.cu"),
    ],
    verbose=False,
)


TRANSFORM_EPS = 1e-6
DEGREE2RAD = torch.pi / 180.0
RAD2DEGREE = 180.0 / torch.pi


def axisangle2mat_cpu(axisangle, degree=False):
    n = axisangle.shape[0]

    # Split the angles and translations
    angles = axisangle[:, :3]
    translations = axisangle[:, 3:]

    if degree:
        angles = angles * DEGREE2RAD

    # Calculate theta^2
    theta2 = torch.sum(angles**2, dim=1)

    # Create an identity matrix to be filled
    mat = torch.eye(3, 4, dtype=torch.float32).unsqueeze(0).repeat(n, 1, 1)

    # Indices where theta2 > TRANSFORM_EPS
    mask = theta2 > TRANSFORM_EPS
    theta = torch.sqrt(theta2[mask])
    normalized_angles = angles[mask] / theta.unsqueeze(1)

    # Calculate sin, cos, and 1 - cos
    s = torch.sin(theta)
    c = torch.cos(theta)
    o_c = 1 - c

    x, y, z = (
        normalized_angles[:, 0],
        normalized_angles[:, 1],
        normalized_angles[:, 2],
    )

    # Calculate rotation matrix components for masked indices
    mat[mask, 0, 0] = c + x * x * o_c
    mat[mask, 0, 1] = x * y * o_c - z * s
    mat[mask, 0, 2] = y * s + x * z * o_c

    mat[mask, 1, 0] = z * s + x * y * o_c
    mat[mask, 1, 1] = c + y * y * o_c
    mat[mask, 1, 2] = -x * s + y * z * o_c

    mat[mask, 2, 0] = -y * s + x * z * o_c
    mat[mask, 2, 1] = x * s + y * z * o_c
    mat[mask, 2, 2] = c + z * z * o_c

    # Fill with the simplified rotation matrix for theta^2 <= TRANSFORM_EPS
    mat[~mask, 0, 0] = 1
    mat[~mask, 0, 1] = -angles[~mask][:, 2]
    mat[~mask, 0, 2] = angles[~mask][:, 1]

    mat[~mask, 1, 0] = angles[~mask][:, 2]
    mat[~mask, 1, 1] = 1
    mat[~mask, 1, 2] = -angles[~mask][:, 0]

    mat[~mask, 2, 0] = -angles[~mask][:, 1]
    mat[~mask, 2, 1] = angles[~mask][:, 0]
    mat[~mask, 2, 2] = 1

    # Assign the translation components
    mat[:, :, 3] = translations

    return mat


def mat2axisangle_cpu(mat, in_degrees=False):
    # Extract the rotation matrices and translation components
    aff = mat[:, :3, :3]
    translations = mat[:, :3, 3]

    # Calculate the trace of the rotation matrices
    trace = aff.diagonal(dim1=1, dim2=2).sum(dim=1)

    # Initialize the quaternion components
    w = torch.zeros_like(trace)
    x = torch.zeros_like(trace)
    y = torch.zeros_like(trace)
    z = torch.zeros_like(trace)

    mask_d2 = aff[:, 2, 2] < TRANSFORM_EPS
    mask_d0_d1 = aff[:, 0, 0] > aff[:, 1, 1]
    mask_d0_nd1 = aff[:, 0, 0] < -aff[:, 1, 1]

    # Case 1: r00 + r11 + r22 + 1 is positive
    s = 2.0 * torch.sqrt(trace + 1.0)
    idx = (~mask_d2) & (~mask_d0_nd1)
    w[idx] = 0.25 * s[idx]
    x[idx] = (aff[idx, 2, 1] - aff[idx, 1, 2]) / s[idx]
    y[idx] = (aff[idx, 0, 2] - aff[idx, 2, 0]) / s[idx]
    z[idx] = (aff[idx, 1, 0] - aff[idx, 0, 1]) / s[idx]

    # Case 2: r00 > r11 and r22 < TRANSFORM_EPSILON
    s = 2.0 * torch.sqrt(aff[:, 0, 0] - aff[:, 1, 1] - aff[:, 2, 2] + 1.0)
    idx = mask_d2 & mask_d0_d1
    w[idx] = (aff[idx, 2, 1] - aff[idx, 1, 2]) / s[idx]
    x[idx] = 0.25 * s[idx]
    y[idx] = (aff[idx, 0, 1] + aff[idx, 1, 0]) / s[idx]
    z[idx] = (aff[idx, 0, 2] + aff[idx, 2, 0]) / s[idx]

    # Case 3: r00 < -r11 and r22 < TRANSFORM_EPSILON
    s = 2.0 * torch.sqrt(aff[:, 1, 1] - aff[:, 0, 0] - aff[:, 2, 2] + 1.0)
    idx = mask_d2 & (~mask_d0_d1)
    w[idx] = (aff[idx, 0, 2] - aff[idx, 2, 0]) / s[idx]
    x[idx] = (aff[idx, 0, 1] + aff[idx, 1, 0]) / s[idx]
    y[idx] = 0.25 * s[idx]
    z[idx] = (aff[idx, 1, 2] + aff[idx, 2, 1]) / s[idx]

    # Case 4: Default case (r22 > TRANSFORM_EPSILON)
    s = 2.0 * torch.sqrt(aff[:, 2, 2] - aff[:, 0, 0] - aff[:, 1, 1] + 1.0)
    idx = (~mask_d2) & mask_d0_nd1
    w[idx] = (aff[idx, 1, 0] - aff[idx, 0, 1]) / s[idx]
    x[idx] = (aff[idx, 0, 2] + aff[idx, 2, 0]) / s[idx]
    y[idx] = (aff[idx, 1, 2] + aff[idx, 2, 1]) / s[idx]
    z[idx] = 0.25 * s[idx]

    # Normalize quaternion if w < 0
    negative_mask = w < 0
    w[negative_mask] *= -1
    x[negative_mask] *= -1
    y[negative_mask] *= -1
    z[negative_mask] *= -1

    # Compute axis-angle
    norm_axis = torch.sqrt(x**2 + y**2 + z**2)
    theta = 2 * torch.atan2(norm_axis, w)
    factor = torch.where(norm_axis > TRANSFORM_EPS, theta / norm_axis, 2.0 / w)

    axis_angle = torch.zeros((aff.shape[0], 6), dtype=aff.dtype, device=aff.device)
    axis_angle[:, 0] = x * factor
    axis_angle[:, 1] = y * factor
    axis_angle[:, 2] = z * factor

    if in_degrees:
        axis_angle[:, :3] *= RAD2DEGREE

    # Append translation components
    axis_angle[:, 3:] = translations

    return axis_angle


class Axisangle2MatFunction(Function):
    @staticmethod
    def forward(ctx, axisangle):
        if axisangle.is_cuda:
            outputs = transform_convert_cuda.axisangle2mat_forward(axisangle)
            mat = outputs[0]
        else:
            mat = axisangle2mat_cpu(axisangle)
        if axisangle.requires_grad:
            ctx.save_for_backward(axisangle)
        return mat

    @staticmethod
    def backward(ctx, grad_mat):
        axisangle = ctx.saved_variables[0]
        outputs = transform_convert_cuda.axisangle2mat_backward(grad_mat, axisangle)
        grad_axisangle = outputs[0]
        return grad_axisangle


class Mat2AxisangleFunction(Function):
    @staticmethod
    def forward(ctx, mat):
        if mat.is_cuda:
            outputs = transform_convert_cuda.mat2axisangle_forward(mat)
            axisangle = outputs[0]
        else:
            axisangle = mat2axisangle_cpu(mat)
        if mat.requires_grad:
            ctx.save_for_backward(mat)
        return axisangle

    @staticmethod
    def backward(ctx, grad_axisangle):
        mat = ctx.saved_variables[0]
        outputs = transform_convert_cuda.mat2axisangle_backward(mat, grad_axisangle)
        grad_mat = outputs[0]
        return grad_mat


def axisangle2mat(axisangle):
    return Axisangle2MatFunction.apply(axisangle)


def mat2axisangle(mat):
    return Mat2AxisangleFunction.apply(mat)
