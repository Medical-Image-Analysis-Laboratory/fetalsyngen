import torch
from fetalsyngen.intensity.seed_sampler import ImageFromSeeds
from fetalsyngen.deformation.affine_nonrigid import SpatialDeformation
from fetalsyngen.augmentation.synthseg_augm import (
    RandResample,
    RandBiasField,
    RandGamma,
    RandNoise,
)
import time
import SimpleITK as sitk
from typing import Iterable
import numpy as np

# TODO: Make different versions of the feta.yaml that work
# with FetalSimpleDataset and FetalSynthDataset as well as
# versions with/withoout transforms, with/without seeds
# and make an example notebook where all is called and illustrated
# time the synthetic generation


class SynthGen:

    def __init__(
        self,
        shape: Iterable[int],
        resolution: Iterable[float],
        image_seed_generator: ImageFromSeeds,
        rand_resample: RandResample,
        rand_biasfield: RandBiasField,
        rand_noise: RandNoise,
        rand_gamma: RandGamma,
        spatial_deform: SpatialDeformation,
        # image_transforms,
        device: str,
    ):
        self.shape = shape
        self.resolution = resolution
        self.intensity_generator = image_seed_generator
        self.spatial_deform = spatial_deform
        # self.image_transforms = image_transforms
        self.resampled = rand_resample
        self.biasfield = rand_biasfield
        self.gamma = rand_gamma
        self.noise = rand_noise
        self.device = device
        # seed loading parameters

    def sample(self, image, segmentation, seeds: torch.Tensor | None):
        generation_params = {}

        # 1. Generate intensity output
        # if seeds are passed, used them to generate the intensity
        # image randomly, otherwise skip this step and use the
        # original image
        start_time = time.time()

        if seeds is not None:
            seeds = self.intensity_generator.load_seeds(seeds)

            output = self.intensity_generator.sample_intensities(seeds)
            intensity_time = time.time() - start_time
            print("Intensity time: ", intensity_time)
            # 0.21s per generation already and ~400mb of GPU
        else:
            if image is None:
                raise ValueError(
                    "If no seeds are passed, an image must be loaded to be used as intensity prior!"
                )
            output = image

        # 2. Spatially deform the inputs
        image, segmentation, output = self.spatial_deform.deform(
            image, segmentation, output
        )
        spatial_defom_time = time.time() - start_time
        print("Spatial deformation time: ", spatial_defom_time)

        # 3. Gamma
        output = self.gamma(output)

        # 4. Bias field
        output = self.biasfield(output)

        # 5. Downsample
        print(self.resolution)
        output, factors = self.resampled(output, self.device, np.array(self.resolution))
        resampling_time = time.time() - start_time
        print("Resampling time: ", resampling_time)
        print(f"Shape: {image.shape}, {segmentation.shape}, {output.shape}")

        # 6. Noise
        output = self.noise(output)

        # 7. Up-sample back
        output = self.resampled.resize_back(output, factors)
        resampling_time = time.time() - start_time
        print("Resampling time: ", resampling_time)
        print(f"Up-scaled shapes: {image.shape}, {segmentation.shape}, {output.shape}")

        # 8. SR-artifacts

        # Apply augmentations

        # 4. Augment the output
        # don't change the image
        # print("Augmentation time: ", time.time() - start_time)
        # print(self.image_transforms)
        # output = self.image_transforms(output)
        # Resampling
        # Bias
        # Noise
        # Blur
        # Contrast - gama
        sitk.WriteImage(sitk.GetImageFromArray(image.cpu().numpy()), "image.nii")
        sitk.WriteImage(sitk.GetImageFromArray(output.cpu().numpy()), "output.nii")
        sitk.WriteImage(
            sitk.GetImageFromArray(segmentation.cpu().numpy()), "segmentation.nii"
        )
        print(f"Total time: {time.time() - start_time}")
        return output, segmentation, image, generation_params

        # # change to n_channels, x, y, z
        # print(f"Shape: {image.shape}, {segmentation.shape}, {output.shape}")
        # print(f"Dtype: {image.dtype}, {segmentation.dtype}, {output.dtype}")
        # print(f"Device: {image.device}, {segmentation.device}, {output.device}")
        # # save image and the segmentation
        # sitk.WriteImage(sitk.GetImageFromArray(output[0].cpu().numpy()), "output.nii")
        # print("Intensity+Deformation time: ", time.time() - start_time)

        # return output, segmentation, image, generation_params
