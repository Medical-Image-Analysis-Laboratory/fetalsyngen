from torch import nn
from torch.autograd import Function
import torch
from torch.utils.cpp_extension import load
import os
from typing import Optional, cast, Sequence
import torch.nn.functional as F

BATCH_SIZE = 64
dirname = os.path.dirname(__file__)

slice_acq_cuda = load(
    "slice_acq_cuda",
    [
        os.path.join(dirname, "slice_acq_cuda.cpp"),
        os.path.join(dirname, "slice_acq_cuda_kernel.cu"),
    ],
    verbose=False,
)


class SliceAcqFunction(Function):
    @staticmethod
    def forward(
        ctx,
        transforms,
        vol,
        vol_mask,
        slices_mask,
        psf,
        slice_shape,
        res_slice,
        need_weight,
        interp_psf,
    ):

        if vol_mask is None:
            vol_mask = torch.empty(0, device=vol.device)
        if slices_mask is None:
            slices_mask = torch.empty(0, device=vol.device)

        # ensure that the input is contiguous
        vol = vol.contiguous()
        vol_mask = vol_mask.contiguous()
        slices_mask = slices_mask.contiguous()

        outputs = slice_acq_cuda.forward(
            transforms,
            vol,
            vol_mask,
            slices_mask,
            psf,
            slice_shape,
            res_slice,
            need_weight,
            interp_psf,
        )
        ctx.save_for_backward(transforms, vol, vol_mask, slices_mask, psf)
        ctx.interp_psf = interp_psf
        ctx.res_slice = res_slice
        ctx.need_weight = need_weight

        if need_weight:
            return outputs[0], outputs[1]
        else:
            return outputs[0]

    @staticmethod
    def backward(ctx, *args):
        if ctx.need_weight:
            assert len(args) == 2
        grad_slices = args[0]
        transforms, vol, vol_mask, slices_mask, psf = ctx.saved_variables
        interp_psf = ctx.interp_psf
        res_slice = ctx.res_slice
        need_vol_grad = ctx.needs_input_grad[1]
        need_transforms_grad = ctx.needs_input_grad[0]
        outputs = slice_acq_cuda.backward(
            transforms,
            vol,
            vol_mask,
            psf,
            grad_slices.contiguous(),
            slices_mask,
            res_slice,
            interp_psf,
            need_vol_grad,
            need_transforms_grad,
        )
        grad_vol, grad_transforms = outputs
        return (
            grad_transforms,
            grad_vol,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class SliceAcqAdjointFunction(Function):
    @staticmethod
    def forward(
        ctx,
        transforms,
        psf,
        slices,
        slices_mask,
        vol_mask,
        vol_shape,
        res_slice,
        interp_psf,
        equalize,
    ):
        if vol_mask is None:
            vol_mask = torch.empty(0, device=slices.device)
        if slices_mask is None:
            slices_mask = torch.empty(0, device=slices.device)

        outputs = slice_acq_cuda.adjoint_forward(
            transforms,
            psf,
            slices,
            slices_mask,
            vol_mask,
            vol_shape,
            res_slice,
            interp_psf,
            equalize,
        )
        vol, vol_weight = outputs
        if equalize:
            ctx.save_for_backward(
                transforms,
                psf,
                slices,
                slices_mask,
                vol_mask,
                vol,
                vol_weight,
            )
        else:
            ctx.save_for_backward(transforms, psf, slices, slices_mask, vol_mask)
        ctx.res_slice = res_slice
        ctx.interp_psf = interp_psf
        ctx.equalize = equalize
        return vol

    @staticmethod
    def backward(ctx, grad_vol):
        res_slice = ctx.res_slice
        interp_psf = ctx.interp_psf
        equalize = ctx.equalize
        if equalize:
            transforms, psf, slices, slices_mask, vol_mask, vol, vol_weight = (
                ctx.saved_variables
            )
        else:
            transforms, psf, slices, slices_mask, vol_mask = ctx.saved_variables
            vol = vol_weight = torch.empty(0)
        need_slices_grad = ctx.needs_input_grad[2]
        need_transforms_grad = ctx.needs_input_grad[0]
        outputs = slice_acq_cuda.adjoint_backward(
            transforms,
            grad_vol,
            vol_weight,
            vol_mask,
            psf,
            slices,
            slices_mask,
            vol,
            res_slice,
            interp_psf,
            equalize,
            need_slices_grad,
            need_transforms_grad,
        )
        grad_slices, grad_transforms = outputs
        return (
            grad_transforms,
            None,
            grad_slices,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def slice_acquisition(
    transforms,
    vol,
    vol_mask,
    slices_mask,
    psf,
    slice_shape,
    res_slice,
    need_weight,
    interp_psf,
):
    if not vol.is_cuda:
        return slice_acquisition_torch(
            transforms,
            vol,
            vol_mask,
            slices_mask,
            psf,
            slice_shape,
            res_slice,
            need_weight,
        )
    else:
        return SliceAcqFunction.apply(
            transforms,
            vol,
            vol_mask,
            slices_mask,
            psf,
            slice_shape,
            res_slice,
            need_weight,
            interp_psf,
        )


def slice_acquisition_adjoint(
    transforms,
    psf,
    slices,
    slices_mask,
    vol_mask,
    vol_shape,
    res_slice,
    interp_psf,
    equalize,
):

    if not slices.is_cuda:
        return slice_acquisition_adjoint_torch(
            transforms,
            psf,
            slices,
            slices_mask,
            vol_mask,
            vol_shape,
            res_slice,
            equalize,
        )
    else:
        return SliceAcqAdjointFunction.apply(
            transforms,
            psf,
            slices,
            slices_mask,
            vol_mask,
            vol_shape,
            res_slice,
            interp_psf,
            equalize,
        )


def xyz_masked_untransformed(mask, shape, res) -> torch.Tensor:
    shape = torch.tensor(shape, dtype=torch.float32)
    kji = torch.nonzero(mask)
    return torch.flip((kji - (shape - 1) / 2) * res, (-1,))


def _construct_slice_coef(
    i, transform, vol_shape, slice_shape, vol_mask, slice_mask, psf, res_slice
):
    transform = transform[None]
    psf_xyz = xyz_masked_untransformed(psf > 0, psf.shape[-3:], 1.0)
    psf_v = psf[psf > 0]
    if slice_mask is not None:
        _slice = slice_mask
    else:
        _slice = torch.ones((1,) + slice_shape, dtype=torch.bool, device=psf.device)
    slice_xyz = xyz_masked_untransformed(_slice, _slice.shape[-3:], res_slice)
    # transformation
    slice_xyz = mat_transform_points(transform, slice_xyz, trans_first=True)
    psf_xyz = mat_transform_points(
        transform, psf_xyz - transform[:, :, -1], trans_first=True
    )
    #
    shift_xyz = (
        torch.tensor(vol_shape[::-1], dtype=psf.dtype, device=psf.device) - 1
    ) / 2.0
    # (n_pixel, n_psf, 3)
    slice_xyz = shift_xyz + psf_xyz.reshape((1, -1, 3)) + slice_xyz.reshape((-1, 1, 3))
    # (n_pixel, n_psf)
    inside_mask = torch.all((slice_xyz > 0) & (slice_xyz < (shift_xyz * 2)), -1)
    # (n_masked, 3)
    slice_xyz = slice_xyz[inside_mask].round().long()
    # (n_masked,)
    slice_id = torch.arange(
        i * slice_shape[0] * slice_shape[1],
        (i + 1) * slice_shape[0] * slice_shape[1],
        dtype=torch.long,
        device=psf.device,
    )
    if slice_mask is not None:
        slice_id = slice_id.view_as(slice_mask)[slice_mask]
    slice_id = slice_id[..., None].expand(-1, psf_v.shape[0])[inside_mask]
    psf_v = psf_v[None].expand(inside_mask.shape[0], -1)[inside_mask]
    volume_id = (
        slice_xyz[:, 0]
        + slice_xyz[:, 1] * vol_shape[2]
        + slice_xyz[:, 2] * (vol_shape[1] * vol_shape[2])
    )
    return slice_id, volume_id, psf_v


def _construct_coef(
    idxs,
    transforms,
    vol_shape,
    slice_shape,
    vol_mask,
    slice_mask,
    psf,
    res_slice,
):
    # if not check_cache(transforms, vol_shape, slice_shape, psf, res_slice):
    #    clean_cache(transforms, vol_shape, slice_shape, psf, res_slice)
    #    print("clean cache")
    # if False and idxs[0] in _cache:
    #     # print("cache")
    #     return _cache[idxs[0]].to(transforms.device)
    # else:

    slice_ids = []
    volume_ids = []
    psf_vs = []
    for i in range(len(idxs)):
        slice_id, volume_id, psf_v = _construct_slice_coef(
            i,
            transforms[idxs[i]],
            vol_shape,
            slice_shape,
            vol_mask,
            slice_mask[idxs[i]] if slice_mask is not None else None,
            psf,
            res_slice,
        )
        slice_ids.append(slice_id)
        volume_ids.append(volume_id)
        psf_vs.append(psf_v)

    slice_id = torch.cat(slice_ids)
    del slice_ids
    volume_id = torch.cat(volume_ids)
    del volume_ids
    ids = torch.stack((slice_id, volume_id), 0)
    del slice_id, volume_id
    psf_v = torch.cat(psf_vs)
    del psf_vs
    coef = torch.sparse_coo_tensor(
        ids,
        psf_v,
        [
            slice_shape[0] * slice_shape[1] * len(idxs),
            vol_shape[0] * vol_shape[1] * vol_shape[2],
        ],
    ).coalesce()
    # _cache[idxs[0]] = coef.cpu()
    return coef


def slice_acquisition_torch(
    transforms: torch.Tensor,
    vol: torch.Tensor,
    vol_mask: Optional[torch.Tensor],
    slices_mask: Optional[torch.Tensor],
    psf: torch.Tensor,
    slice_shape: Sequence,
    res_slice: float,
    need_weight: bool,
):
    slice_shape = tuple(slice_shape)
    global BATCH_SIZE
    if psf.numel() == 1 and need_weight == False:
        return slice_acquisition_no_psf_torch(
            transforms, vol, vol_mask, slices_mask, slice_shape, res_slice
        )
    if vol_mask is not None:
        vol = vol * vol_mask
    vol_shape = vol.shape[-3:]
    _slices = []
    _weights = []
    i = 0
    while i < transforms.shape[0]:
        succ = False
        try:
            coef = _construct_coef(
                list(range(i, min(i + BATCH_SIZE, transforms.shape[0]))),
                transforms,
                vol_shape,
                slice_shape,
                vol_mask,
                slices_mask,
                psf,
                res_slice,
            )
            s = torch.mv(coef, vol.view(-1)).to_dense().reshape((-1, 1) + slice_shape)
            weight = torch.sparse.sum(coef, 1).to_dense().reshape_as(s)
            del coef
            succ = True
        except RuntimeError as e:
            if "out of memory" in str(e) and BATCH_SIZE > 0:
                print("OOM, reduce batch size")
                BATCH_SIZE = BATCH_SIZE // 2
                torch.cuda.empty_cache()
            else:
                raise e
        if succ:
            _slices.append(s)
            _weights.append(weight)
            i += BATCH_SIZE

    slices = torch.cat(_slices)
    weights = torch.cat(_weights)
    m = weights > 1e-2
    slices[m] = slices[m] / weights[m]
    if slices_mask is not None:
        slices = slices * slices_mask
    if need_weight:
        return slices, weights
    else:
        return slices


def mat_transform_points(
    mat: torch.Tensor, x: torch.Tensor, trans_first: bool
) -> torch.Tensor:
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


def slice_acquisition_no_psf_torch(
    transforms: torch.Tensor,
    vol: torch.Tensor,
    vol_mask: Optional[torch.Tensor],
    slices_mask: Optional[torch.Tensor],
    slice_shape: Sequence,
    res_slice: float,
) -> torch.Tensor:
    slice_shape = tuple(slice_shape)
    device = transforms.device
    _slice = torch.ones((1,) + slice_shape, dtype=torch.bool, device=device)
    slice_xyz = xyz_masked_untransformed(_slice, _slice.shape[-3:], res_slice)
    # transformation
    slice_xyz = mat_transform_points(
        transforms[:, None], slice_xyz[None], trans_first=True
    ).view((transforms.shape[0], 1) + slice_shape + (3,))

    output_slices = torch.zeros_like(slice_xyz[..., 0])

    if slices_mask is not None:
        masked_xyz = slice_xyz[slices_mask]
    else:
        masked_xyz = slice_xyz

    # shape = xyz.shape[:-1]
    masked_xyz = masked_xyz / (
        (torch.tensor(vol.shape[-3:][::-1], dtype=masked_xyz.dtype, device=device) - 1)
        / 2
    )
    if vol_mask is not None:
        vol = vol * vol_mask
    masked_v = F.grid_sample(vol, masked_xyz.view(1, 1, 1, -1, 3), align_corners=True)
    if slices_mask is not None:
        output_slices[slices_mask] = masked_v
    else:
        output_slices = masked_v.reshape((transforms.shape[0], 1) + slice_shape)
    return output_slices


def slice_acquisition_adjoint_torch(
    transforms: torch.Tensor,
    psf: torch.Tensor,
    slices: torch.Tensor,
    slices_mask: Optional[torch.Tensor],
    vol_mask: Optional[torch.Tensor],
    vol_shape: Sequence,
    res_slice: float,
    equalize: bool,
):
    vol_shape = tuple(vol_shape)
    global BATCH_SIZE
    if slices_mask is not None:
        slices = slices * slices_mask
    vol = None
    weight = None
    slice_shape = slices.shape[-2:]
    i = 0
    while i < transforms.shape[0]:
        succ = False
        try:
            coef = _construct_coef(
                list(range(i, min(i + BATCH_SIZE, transforms.shape[0]))),
                transforms,
                vol_shape,
                slice_shape,
                vol_mask,
                slices_mask,
                psf,
                res_slice,
            ).t()
            v = torch.mv(coef, slices[i : i + BATCH_SIZE].view(-1))
            if equalize:
                w = torch.sparse.sum(coef, 1)
            del coef
            succ = True
        except RuntimeError as e:
            if "out of memory" in str(e) and BATCH_SIZE > 0:
                print("OOM, reduce batch size")
                BATCH_SIZE = BATCH_SIZE // 2
                torch.cuda.empty_cache()
            else:
                raise e
        if succ:
            if vol is None:
                vol = v
            else:
                vol += v
            if equalize:
                if weight is None:
                    weight = w
                else:
                    weight += w
            i += BATCH_SIZE
    vol = cast(torch.Tensor, vol)
    vol = vol.to_dense().reshape((1, 1) + vol_shape)
    if equalize:
        weight = cast(torch.Tensor, weight)
        weight = weight.to_dense().reshape_as(vol)
        m = weight > 1e-2
        vol[m] = vol[m] / weight[m]
    if vol_mask is not None:
        vol = vol * vol_mask
    return vol
