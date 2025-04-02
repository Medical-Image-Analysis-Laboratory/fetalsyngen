import numpy as np
from SimpleITK import ReadImage, GetArrayFromImage
from monai.data import MetaTensor
from torch import from_numpy, Tensor
from pathlib import Path
import SimpleITK as sitk


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

    def __call__(
        self,
        img_path: str | Path,
        as_meta=True,
        interp: str = "nearest",
        resolution: list[int] | int | None = None,
        spatial_size: list[int] | int | None = None,
    ) -> Tensor | MetaTensor:
        """Reads an image from a path and returns it as a MetaTensor
        if as_meta is True. Otherwise, returns the image as a tensor.

        Args:
            img_path: Path to the image.
            as_meta: Defaults to True.
            interp: Interpolation method to use. Should be one of
                "nearest", "linear". Defaults to "nearest".

        Returns:
            torch.Tensor | monai.data.MetaTensor
        """
        assert interp in ["nearest", "linear"]
        interp = sitk.sitkNearestNeighbor if interp == "nearest" else sitk.sitkLinear

        if isinstance(img_path, Path):
            img_path = str(img_path)
        img = ReadImage(img_path)

        # resample image if needed
        if resolution is not None:
            resolution = (
                resolution if isinstance(resolution, list) else [resolution] * 3
            )
            img = self.resample_image(img, resolution, interp)

        # crop or pad image if needed
        if spatial_size is not None:
            spatial_size = (
                spatial_size if isinstance(spatial_size, list) else [spatial_size] * 3
            )
            img = self.crop_or_pad_image(img, spatial_size)

        affine = Tensor(self.make_affine_from_sitk(img))
        img_data = GetArrayFromImage(img)

        # convert to LPS to match nibabel loading order
        img_data = from_numpy(img_data).permute(2, 1, 0)

        # print shape
        # print(f"Image shape: {img_data.shape}")
        if as_meta:
            return MetaTensor(x=img_data, affine=affine)
        else:
            return img_data

    @staticmethod
    def resample_image(
        image, new_spacing=[1.0, 1.0, 1.0], interpolator=sitk.sitkNearestNeighbor
    ):
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        new_size = [
            int(round(osz * ospc / nspc))
            for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampled_image = resampler.Execute(image)

        return resampled_image

    @staticmethod
    def crop_or_pad_image(image, target_size=[256, 256, 256], pad_value=0):
        current_size = image.GetSize()

        # Calculate padding/cropping for each axis
        size_diff = [t - c for t, c in zip(target_size, current_size)]

        lower_pad = [max(sd // 2, 0) for sd in size_diff]
        upper_pad = [max(sd - lp, 0) for sd, lp in zip(size_diff, lower_pad)]

        lower_crop = [max(-sd // 2, 0) for sd in size_diff]
        upper_crop = [max(-sd - lc, 0) for sd, lc in zip(size_diff, lower_crop)]

        # First crop if necessary
        cropped_image = sitk.Crop(image, lower_crop, upper_crop)

        # Then pad if necessary
        padded_image = sitk.ConstantPad(cropped_image, lower_pad, upper_pad, pad_value)

        return padded_image
