import torch
from fetalsyngen.generator.intensity.rand_gmm import ImageFromSeeds
from fetalsyngen.generator.deformation.affine_nonrigid import SpatialDeformation
from fetalsyngen.generator.augmentation.synthseg import (
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


# TODO: refactor argument names here
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
        device: str,
    ):
        """
        Initialize the model with the given parameters.

        Args:
            shape (Iterable[int]): The shape of the generated images.
            resolution (Iterable[float]): The resolution of the generated images.
            image_seed_generator (ImageFromSeeds): The generator for creating images from seeds.
            rand_resample (RandResample): The random resampling transformation.
            rand_biasfield (RandBiasField): The random bias field transformation.
            rand_noise (RandNoise): The random noise transformation.
            rand_gamma (RandGamma): The random gamma transformation.
            spatial_deform (SpatialDeformation): The spatial deformation transformation.
            device (str): The device to run the model on (e.g., 'cpu' or 'cuda').
        """
        self.shape = shape
        self.resolution = resolution
        self.intensity_generator = image_seed_generator
        self.spatial_deform = spatial_deform
        self.resampled = rand_resample
        self.biasfield = rand_biasfield
        self.gamma = rand_gamma
        self.noise = rand_noise
        self.device = device

    def sample(self, image, segmentation, seeds: torch.Tensor | None):
        synth_params = {}

        # 1. Generate intensity output.
        if seeds is not None:
            seeds, selected_seeds = self.intensity_generator.load_seeds(seeds)
            output, seed_intensities = self.intensity_generator.sample_intensities(
                seeds, self.device
            )
        else:
            if image is None:
                raise ValueError(
                    "If no seeds are passed, an image must be loaded to be used as intensity prior!"
                )
            output = image
            selected_seeds = {}
            seed_intensities = {}

        # ensure that tensors are on the same device
        output = output.to(self.device)
        segmentation = segmentation.to(self.device)
        image = image.to(self.device) if image is not None else None

        # 2. Spatially deform the data
        image, segmentation, output, deform_params = self.spatial_deform.deform(
            image, segmentation, output
        )

        # 3. Gamma contrast transformation
        output, gamma_params = self.gamma(output, self.device)

        # 4. Bias field corruption
        output, bf_params = self.biasfield(output, self.device)

        # 5. Downsample to simulate lower reconstruction resolution
        output, factors, resample_params = self.resampled(
            output, np.array(self.resolution), self.device
        )

        # 6. Noise corruption
        output, noise_params = self.noise(output, self.device)

        # 7. Up-sample back to the original resolution/shape
        output = self.resampled.resize_back(output, factors)

        # 8. Induce SR-artifacts
        # TODO: Thomas add the SR artifacts here

        # 9. Aggregete the synth params
        synth_params.update(
            {
                "selected_seeds": selected_seeds,
                "seed_intensities": seed_intensities,
                "deform_params": deform_params,
                "gamma_params": gamma_params,
                "bf_params": bf_params,
                "resample_params": resample_params,
                "noise_params": noise_params,
            }
        )
        return output, segmentation, image, synth_params
