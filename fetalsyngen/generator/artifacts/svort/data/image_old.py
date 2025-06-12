"""
Define the stacks, slices and volumes.
This is based on the original code of NeSVoR, so this might not be totally appropriate for our use,
as this supposes a coordinate-based approach, while we directly model images.
"""

from __future__ import annotations
import os
from typing import (
    Dict,
    Optional,
    Union,
    List,
    Tuple,
    Sequence,
    cast,
    Collection,
)
import numpy as np
import torch
import torch.nn.functional as F
from fetalsyngen.generator.artifacts.svort.transform import (
    RigidTransform,
    affine2transformation,
    compare_resolution_affine,
    transformation2affine,
    init_stack_transform,
    init_zero_transform,
    transform_points,
)
from fetalsyngen.generator.artifacts.svort.definitions import (
    PathType,
    DeviceType,
)
import nibabel as nib
from fetalsyngen.generator.artifacts.svort.data.utils import meshgrid, resample


class _Data(object):
    def __init__(
        self,
        data: torch.Tensor,
        mask: Optional[torch.Tensor],
        transformation: Optional[RigidTransform],
    ) -> None:
        if mask is None:
            mask = torch.ones_like(data, dtype=torch.bool)
        if transformation is None:
            transformation = init_zero_transform(1, data.device)
        self.data = data
        self.mask = mask
        self.transformation = transformation

    def check_data(self, value) -> None:
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("Data must be Tensor!")

    def check_mask(self, value) -> None:
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("Mask must be Tensor!")
        if value.shape != self.shape:
            raise RuntimeError("Mask has a shape different from image!")
        if value.dtype != torch.bool:
            raise RuntimeError("Mask must be bool!")
        if value.device != self.device:
            raise RuntimeError("The device of mask is different!")

    def check_transformation(self, value) -> None:
        if not isinstance(value, RigidTransform):
            raise RuntimeError("Transformation must be RigidTransform")
        if value.device != self.device:
            raise RuntimeError("The device of transformation must be the same as data!")

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @data.setter
    def data(self, value: torch.Tensor) -> None:
        self.check_data(value)
        self._data = value

    @property
    def mask(self) -> torch.Tensor:
        return self._mask

    @mask.setter
    def mask(self, value: torch.Tensor) -> None:
        self.check_mask(value)
        self._mask = value

    @property
    def transformation(self) -> RigidTransform:
        return self._transformation

    @transformation.setter
    def transformation(self, value: RigidTransform) -> None:
        self.check_transformation(value)
        self._transformation = value

    @property
    def device(self) -> DeviceType:
        return self.data.device

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    def clone(self, *, zero: bool = False, deep: bool = True) -> _Data:
        raise NotImplementedError()

    def _clone_dict(self, zero: bool = False, deep: bool = True) -> Dict:
        data = self.data
        mask = self.mask
        transformation = self.transformation
        if zero:
            data = torch.zeros_like(data)
            mask = torch.zeros_like(mask)
        elif deep:
            data = data.clone()
            mask = mask.clone()
        if deep:
            transformation = transformation.clone()
        return {
            "data": data,
            "mask": mask,
            "transformation": self.transformation,
        }


class Image(_Data):
    def __init__(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        transformation: Optional[RigidTransform] = None,
        resolution_x: Union[float, torch.Tensor] = 1.0,
        resolution_y: Union[float, torch.Tensor, None] = None,
        resolution_z: Union[float, torch.Tensor, None] = None,
    ) -> None:
        super().__init__(image, mask, transformation)
        if resolution_y is None:
            resolution_y = resolution_x
        if resolution_z is None:
            resolution_z = resolution_x
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_z = resolution_z

    def check_data(self, value) -> None:
        super().check_data(value)
        if value.ndim != 3:
            raise RuntimeError("The dimension of image must be 3!")

    def check_transformation(self, value) -> None:
        super().check_transformation(value)
        if len(value) != 1:
            raise RuntimeError("The len of transformation must be 1!")

    @property
    def image(self) -> torch.Tensor:
        return self.data

    @image.setter
    def image(self, value: torch.Tensor) -> None:
        self.data = value

    def _clone_dict(self, zero: bool = False, deep: bool = True) -> Dict:
        d = super()._clone_dict(zero, deep)
        d["resolution_x"] = float(self.resolution_x)
        d["resolution_y"] = float(self.resolution_y)
        d["resolution_z"] = float(self.resolution_z)
        d["image"] = d.pop("data")
        return d

    @property
    def shape_xyz(self) -> torch.Tensor:
        return torch.tensor(self.image.shape[::-1], device=self.image.device)

    @property
    def resolution_xyz(self) -> torch.Tensor:
        return torch.tensor(
            [self.resolution_x, self.resolution_y, self.resolution_z],
            device=self.image.device,
        )

    def save(self, path: PathType, masked=True) -> None:
        affine = transformation2affine(
            self.image,
            self.transformation,
            float(self.resolution_x),
            float(self.resolution_y),
            float(self.resolution_z),
        )
        if masked:
            output_volume = self.image * self.mask.to(self.image.dtype)
        else:
            output_volume = self.image
        save_nii_volume(path, output_volume, affine)

    def save_mask(self, path: PathType) -> None:
        affine = transformation2affine(
            self.image,
            self.transformation,
            float(self.resolution_x),
            float(self.resolution_y),
            float(self.resolution_z),
        )
        output_volume = self.mask.to(self.image.dtype)
        save_nii_volume(path, output_volume, affine)

    @property
    def xyz_masked_untransformed(self) -> torch.Tensor:
        kji = torch.flip(torch.nonzero(self.mask), (-1,))
        return (kji - (self.shape_xyz - 1) / 2) * self.resolution_xyz

    @property
    def v_masked(self) -> torch.Tensor:
        return self.image[self.mask]

    def rescale(self, intensity_mean: Union[float, torch.Tensor], masked: bool = True) -> None:
        if masked:
            scale_factor = intensity_mean / self.image[self.mask].mean()
        else:
            scale_factor = intensity_mean / self.image.mean()
        self.image *= scale_factor

    @staticmethod
    def like(
        old: Image,
        image: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        deep: bool = True,
    ) -> Image:
        if image is None:
            image = old.image.clone() if deep else old.image
        if mask is None:
            mask = old.mask.clone() if deep else old.mask
        transformation = old.transformation.clone() if deep else old.transformation
        return old.__class__(
            image=image,
            mask=mask,
            transformation=transformation,
            resolution_x=old.resolution_x,
            resolution_y=old.resolution_y,
            resolution_z=old.resolution_z,
        )


class Slice(Image):
    def check_data(self, value) -> None:
        super().check_data(value)
        if value.shape[0] != 1:
            raise RuntimeError("The shape of a slice must be (1, H, W)!")

    def clone(self, *, zero: bool = False, deep: bool = True) -> Slice:
        return Slice(
            **self._clone_dict(zero, deep),
        )

    def resample(
        self,
        resolution_new: Union[float, Sequence],
    ) -> Slice:
        if isinstance(resolution_new, float) or len(resolution_new) == 1:
            resolution_new = [resolution_new, resolution_new]

        if len(resolution_new) == 3:
            resolution_z_new = resolution_new[-1]
            resolution_new = resolution_new[:-1]
        else:
            resolution_z_new = self.resolution_z

        image = resample(
            self.image[None],
            (self.resolution_x, self.resolution_y),
            resolution_new,
        )[0]
        mask = (
            resample(
                self.mask[None].float(),
                (self.resolution_x, self.resolution_y),
                resolution_new,
            )[0]
            > 0
        )

        new_slice = cast(Slice, Slice.like(self, image, mask, deep=True))
        new_slice.resolution_z = resolution_z_new

        return new_slice


class Volume(Image):
    def sample_points(self, xyz: torch.Tensor) -> torch.Tensor:
        shape = xyz.shape[:-1]
        xyz = transform_points(self.transformation.inv(), xyz.view(-1, 3))
        xyz = xyz / ((self.shape_xyz - 1) * self.resolution_xyz / 2)
        return F.grid_sample(
            self.image[None, None],
            xyz.view(1, 1, 1, -1, 3),
            align_corners=True,
        ).view(shape)

    def resample(
        self,
        resolution_new: Optional[Union[float, torch.Tensor]],
        transformation_new: Optional[RigidTransform],
    ) -> Volume:
        if transformation_new is None:
            transformation_new = self.transformation
        R = transformation_new.matrix()[0, :3, :3]
        dtype = R.dtype
        device = R.device
        if resolution_new is None:
            resolution_new = self.resolution_xyz
        elif isinstance(resolution_new, float) or resolution_new.numel == 1:
            resolution_new = torch.tensor([resolution_new] * 3, dtype=dtype, device=device)

        xyz = self.xyz_masked
        # new rotation
        xyz = torch.matmul(torch.inverse(R), xyz.view(-1, 3, 1))[..., 0]

        xyz_min = xyz.amin(0) - resolution_new * 10
        xyz_max = xyz.amax(0) + resolution_new * 10
        shape_xyz = ((xyz_max - xyz_min) / resolution_new).ceil().long()

        mat = torch.zeros((1, 3, 4), dtype=R.dtype, device=R.device)
        mat[0, :, :3] = R
        mat[0, :, -1] = xyz_min + (shape_xyz - 1) / 2 * resolution_new

        xyz = meshgrid(shape_xyz, resolution_new, xyz_min, device, True)

        xyz = torch.matmul(R, xyz[..., None])[..., 0]

        v = self.sample_points(xyz)

        return Volume(
            v,
            v > 0,
            RigidTransform(mat, trans_first=True),
            resolution_new[0].item(),
            resolution_new[1].item(),
            resolution_new[2].item(),
        )

    def clone(self, *, zero: bool = False, deep: bool = True) -> Volume:
        return Volume(**self._clone_dict(zero))

    @staticmethod
    def zeros(
        shape: Tuple,
        resolution_x,
        resolution_y=None,
        resolution_z=None,
        device: DeviceType = None,
    ) -> Volume:
        image = torch.zeros(shape, dtype=torch.float32, device=device)
        mask = torch.ones_like(image, dtype=torch.bool)
        return Volume(
            image,
            mask,
            transformation=None,
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            resolution_z=resolution_z,
        )


class Stack(_Data):
    def __init__(
        self,
        slices: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        transformation: Optional[RigidTransform] = None,
        resolution_x: float = 1.0,
        resolution_y: Optional[float] = None,
        thickness: Optional[float] = None,
        gap: Optional[float] = None,
        name: str = "",
    ) -> None:
        if resolution_y is None:
            resolution_y = resolution_x
        if thickness is None:
            thickness = gap if gap is not None else resolution_x
        if gap is None:
            gap = thickness
        if transformation is None:
            transformation = init_stack_transform(slices.shape[0], gap, slices.device)
        super().__init__(slices, mask, transformation)
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.thickness = thickness
        self.gap = gap
        self.name = name

    def check_data(self, value) -> None:
        super().check_data(value)
        if value.ndim != 4:
            raise RuntimeError("Stack must be 4D data")
        if value.shape[1] != 1:
            raise RuntimeError("Stack must has shape (N, 1, H, W)")

    def check_transformation(self, value) -> None:
        super().check_transformation(value)
        if len(value) != self.slices.shape[0]:
            raise RuntimeError("The number of transformatons is not equal to the number of slices!")

    @property
    def slices(self) -> torch.Tensor:
        return self.data

    @slices.setter
    def slices(self, value: torch.Tensor) -> None:
        self.data = value

    def __len__(self) -> int:
        return self.slices.shape[0]

    def __getitem__(self, idx):
        slices = self.slices[idx]
        masks = self.mask[idx]
        transformation = self.transformation[idx]
        if slices.ndim < self.slices.ndim:
            return Slice(
                slices,
                masks,
                transformation,
                self.resolution_x,
                self.resolution_y,
                self.thickness,
            )
        else:
            return [
                Slice(
                    slices[i],
                    masks[i],
                    transformation[i],
                    self.resolution_x,
                    self.resolution_y,
                    self.thickness,
                )
                for i in range(len(transformation))
            ]

    def get_substack(self, idx_from=None, idx_to=None, /) -> Stack:
        if idx_to is None:
            slices = self.slices[idx_from]
            masks = self.mask[idx_from]
            transformation = self.transformation[idx_from]
        else:
            slices = self.slices[idx_from:idx_to]
            masks = self.mask[idx_from:idx_to]
            transformation = self.transformation[idx_from:idx_to]
        return Stack(
            slices,
            masks,
            transformation,
            self.resolution_x,
            self.resolution_y,
            self.thickness,
            self.gap,
            self.name,
        )

    def get_mask_volume(self) -> Volume:
        mask = self.mask.squeeze(1).clone()
        return Volume(
            image=mask.float(),
            mask=mask > 0,
            transformation=self.transformation.mean(),
            resolution_x=self.resolution_x,
            resolution_y=self.resolution_y,
            resolution_z=self.gap,
        )

    def get_volume(self, copy=True) -> Volume:
        image = self.slices.squeeze(1)
        mask = self.mask.squeeze(1)
        if copy:
            image = image.clone()
            mask = mask.clone()
        return Volume(
            image=image,
            mask=mask,
            transformation=self.transformation.mean(),
            resolution_x=self.resolution_x,
            resolution_y=self.resolution_y,
            resolution_z=self.gap,
        )

    def apply_volume_mask(self, mask: Volume) -> None:
        for i in range(len(self)):
            s = self[i]
            assign_mask = self.mask[i].clone()
            self.mask[i][assign_mask] = mask.sample_points(s.xyz_masked) > 0

    def _clone_dict(self, zero: bool = False, deep: bool = True) -> Dict:
        d = super()._clone_dict(zero, deep)
        d["slices"] = d.pop("data")
        d["resolution_x"] = float(self.resolution_x)
        d["resolution_y"] = float(self.resolution_y)
        d["thickness"] = float(self.thickness)
        d["gap"] = float(self.gap)
        d["name"] = self.name
        return d

    def clone(self, *, zero: bool = False, deep: bool = True) -> Stack:
        return Stack(**self._clone_dict(zero, deep))

    @staticmethod
    def like(
        stack: Stack,
        slices: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        deep: bool = True,
    ) -> Stack:
        if slices is None:
            slices = stack.slices.clone() if deep else stack.slices
        if mask is None:
            mask = stack.mask.clone() if deep else stack.mask
        transformation = stack.transformation.clone() if deep else stack.transformation
        return Stack(
            slices=slices,
            mask=mask,
            transformation=transformation,
            resolution_x=stack.resolution_x,
            resolution_y=stack.resolution_y,
            thickness=stack.thickness,
            gap=stack.gap,
        )

    @staticmethod
    def pad_stacks(stacks: List) -> List:
        size_max = max([max(s.shape[-2:]) for s in stacks])
        lists_pad = []
        for s in stacks:
            if s.shape[-1] < size_max or s.shape[-2] < size_max:
                dx1 = (size_max - s.shape[-1]) // 2
                dx2 = (size_max - s.shape[-1]) - dx1
                dy1 = (size_max - s.shape[-2]) // 2
                dy2 = (size_max - s.shape[-2]) - dy1
                data = F.pad(s.data, (dx1, dx2, dy1, dy2))
                mask = F.pad(s.mask, (dx1, dx2, dy1, dy2))
            else:
                data = s.data
                mask = s.mask
            lists_pad.append(s.__class__.like(s, data, mask, deep=False))
        return lists_pad

    @staticmethod
    def cat(inputs: List) -> Stack:
        data = []
        mask = []
        transformation = []
        for i, inp in enumerate(inputs):
            if isinstance(inp, Slice):
                data.append(inp.image[None])
                mask.append(inp.mask[None])
                transformation.append(inp.transformation)
                if i == 0:
                    resolution_x = float(inp.resolution_x)
                    resolution_y = float(inp.resolution_y)
                    thickness = float(inp.resolution_z)
                    gap = float(inp.resolution_z)
            elif isinstance(inp, Stack):
                data.append(inp.slices)
                mask.append(inp.mask)
                transformation.append(inp.transformation)
                if i == 0:
                    resolution_x = inp.resolution_x
                    resolution_y = inp.resolution_y
                    thickness = inp.thickness
                    gap = inp.gap
            else:
                raise TypeError("unkonwn type!")

        return Stack(
            slices=torch.cat(data, 0),
            mask=torch.cat(mask, 0),
            transformation=RigidTransform.cat(transformation),
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            thickness=thickness,
            gap=gap,
        )

    def init_stack_transform(self) -> RigidTransform:
        return init_stack_transform(len(self), self.gap, self.device)


def save_volume(fname, data, res):
    data = data[0, 0].detach().cpu().numpy().transpose(2, 1, 0)
    w, h, d = data.shape
    affine = np.eye(4)
    affine[:3, :3] *= res
    affine[:3, -1] = -np.array([(w - 1) / 2.0, (h - 1) / 2.0, (d - 1) / 2.0]) * res
    img = nib.Nifti1Image(data, affine)
    img.header.set_xyzt_units(2)
    img.header.set_qform(affine, code="aligned")
    img.header.set_sform(affine, code="scanner")
    nib.save(img, fname)


def save_nii_volume(
    path: PathType,
    volume: Union[torch.Tensor, np.ndarray],
    affine: Optional[Union[torch.Tensor, np.ndarray]],
) -> None:
    assert len(volume.shape) == 3 or (len(volume.shape) == 4 and volume.shape[1] == 1)
    if len(volume.shape) == 4:
        volume = volume.squeeze(1)
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy().transpose(2, 1, 0)
    else:
        volume = volume.transpose(2, 1, 0)
    if isinstance(affine, torch.Tensor):
        affine = affine.detach().cpu().numpy()
    if affine is None:
        affine = np.eye(4)
    if volume.dtype == bool and isinstance(volume, np.ndarray):  # bool type is not supported
        volume = volume.astype(np.int16)
    img = nib.nifti1.Nifti1Image(volume, affine)
    img.header.set_xyzt_units(2)
    img.header.set_qform(affine, code="aligned")
    img.header.set_sform(affine, code="scanner")
    nib.save(img, os.fspath(path))


def load_nii_volume(
    path: PathType,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = nib.load(os.fspath(path))

    dim = img.header["dim"]
    assert dim[0] == 3 or (dim[0] > 3 and all(d == 1 for d in dim[4:])), (
        "Expect a 3D volume but the input is %dD" % dim[0]
    )

    volume = img.get_fdata().astype(np.float32)
    while volume.ndim > 3:
        volume = volume.squeeze(-1)
    volume = volume.transpose(2, 1, 0)

    resolutions = img.header["pixdim"][1:4]

    affine = img.affine
    if np.any(np.isnan(affine)):
        affine = img.get_qform()

    return volume, resolutions, affine


def load_stack(
    path_vol: PathType,
    path_mask: Optional[PathType] = None,
    device: DeviceType = torch.device("cpu"),
) -> Stack:
    slices, resolutions, affine = load_nii_volume(path_vol)
    if path_mask is None:
        mask = np.ones_like(slices, dtype=bool)
    else:
        mask, resolutions_m, affine_m = load_nii_volume(path_mask)
        mask = mask > 0
        if not compare_resolution_affine(
            resolutions,
            affine,
            resolutions_m,
            affine_m,
            slices.shape,
            mask.shape,
        ):
            raise Exception(
                "Error: the sizes/resolutions/affine transformations of the input stack and stack mask do not match!"
            )

    slices_tensor = torch.tensor(slices, device=device)
    mask_tensor = torch.tensor(mask, device=device)

    slices_tensor, mask_tensor, transformation = affine2transformation(
        slices_tensor, mask_tensor, resolutions, affine
    )

    return Stack(
        slices=slices_tensor.unsqueeze(1),
        mask=mask_tensor.unsqueeze(1),
        transformation=transformation,
        resolution_x=resolutions[0],
        resolution_y=resolutions[1],
        thickness=resolutions[2],
        gap=resolutions[2],
        name=str(path_vol),
    )
