import numpy as np
from SimpleITK import ReadImage, GetArrayFromImage
from monai.data import MetaTensor
from torch import from_numpy, Tensor
from pathlib import Path


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

        # convert to LPS to match nibabel loading order
        img_data = from_numpy(img_data).permute(2, 1, 0)
        if as_meta:
            return MetaTensor(x=img_data, affine=affine)
        else:
            return img_data
