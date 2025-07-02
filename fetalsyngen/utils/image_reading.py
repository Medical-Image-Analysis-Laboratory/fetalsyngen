import numpy as np
from SimpleITK import ReadImage, GetArrayFromImage
from monai.data import MetaTensor
from torch import from_numpy, Tensor
from pathlib import Path
import torchio as tio


class SimpleITKReader:
    @staticmethod
    def make_affine_from_sitk(sitk_img):
        """Get affine transform in LPS (niabbel) order from SimpleITK image."""
        if sitk_img.GetDepth() <= 0:
            return np.eye(4)

        rot = [
            sitk_img.TransformContinuousIndexToPhysicalPoint(p)
            for p in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0))
        ]
        rot = np.array(rot)
        affine = np.concatenate(
            [
                np.concatenate([rot[0:3] - rot[3:], rot[3:]], axis=0),
                [[0.0], [0.0], [0.0], [1.0]],
            ],
            axis=1,
        )
        affine = np.transpose(affine)
        # convert to RAS to match nibabel
        affine = np.matmul(np.diag([-1.0, -1.0, 1.0, 1.0]), affine)
        return affine

    def __call__(self, img_path: str | Path, as_meta=True) -> Tensor | MetaTensor:
        """Reads an image from a path and returns it as a MetaTensor
        if as_meta is True. Otherwise, returns the image as a tensor.

        Args:
            img_path: Path to the image.
            as_meta: Defaults to True.

        Returns:
            torch.Tensor | monai.data.MetaTensor
        """
        if isinstance(img_path, Path):
            img_path = str(img_path)
        img = ReadImage(img_path)

        affine = Tensor(self.make_affine_from_sitk(img))
        img_data = GetArrayFromImage(img)

        # convert to RAS to match nibabel loading order
        img_data = from_numpy(img_data).permute(2, 1, 0)
        if as_meta:
            return MetaTensor(x=img_data, affine=affine)
        else:
            return img_data


class TorchIOReader:
    def __call__(self, img_path: str | Path, type: str) -> Tensor | MetaTensor:
        """Reads an image from a path either as an
        intensity image or a label map, returning it as a tio.Image.

        Args:
            img_path: Path to the image.
            type: Type of the image ('intensity' or 'label').

        Returns:
            torchio.Image
        """

        if type not in ["intensity", "label"]:
            raise ValueError("Type must be either 'intensity' or 'label'.")

        if type == "intensity":
            img = tio.ScalarImage(img_path)
        elif type == "label":
            img = tio.LabelMap(img_path)
        return img


# class TorchIOReader:
#     @staticmethod
#     def make_affine_from_sitk(sitk_img):
#         """Get affine transform in LPS (niabbel) order from SimpleITK image."""
#         if sitk_img.GetDepth() <= 0:
#             return np.eye(4)

#         rot = [
#             sitk_img.TransformContinuousIndexToPhysicalPoint(p)
#             for p in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0))
#         ]
#         rot = np.array(rot)
#         affine = np.concatenate(
#             [
#                 np.concatenate([rot[0:3] - rot[3:], rot[3:]], axis=0),
#                 [[0.0], [0.0], [0.0], [1.0]],
#             ],
#             axis=1,
#         )
#         affine = np.transpose(affine)
#         # convert to RAS to match nibabel
#         affine = np.matmul(np.diag([-1.0, -1.0, 1.0, 1.0]), affine)
#         return affine

#     def __call__(self, img_path: str | Path, type: str) -> tio.Image:
#         """Reads an image from a path and returns it as a TorchIO Image.

#         Args:
#             img_path: Path to the image.
#             as_meta: Defaults to True.

#         Returns:
#             torch.Tensor | monai.data.MetaTensor
#         """

#         img = ReadImage(img_path)

#         affine = Tensor(self.make_affine_from_sitk(img))
#         img_data = GetArrayFromImage(img)

#         # convert to RAS to match nibabel loading order
#         img_data = from_numpy(img_data).permute(2, 1, 0).unsqueeze(0)
#         if type == "intensity":
#             return tio.ScalarImage(tensor=img_data, affine=affine)
#         elif type == "label":
#             return tio.LabelMap(tensor=img_data, affine=affine)
#         else:
#             raise ValueError("Type must be either 'intensity' or 'label'.")
