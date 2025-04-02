import torch
from fetalsyngen.generator.intensity.rand_gmm import ImageFromSeeds
from fetalsyngen.generator.deformation.affine_nonrigid import (
    SpatialDeformation,
)
from fetalsyngen.generator.augmentation.synthseg import (
    RandResample,
    RandBiasField,
    RandGamma,
    RandNoise,
)
from typing import Iterable
import numpy as np
from fetalsyngen.generator.artifacts.simulate_reco import (
    Scanner,
    PSFReconstructor,
)
from fetalsyngen.generator.augmentation.artifacts import (
    SimulatedBoundaries,
    StructNoise,
    SimulateMotion,
    BlurCortex,
    StackSampler,
)
from fetalsyngen.generator.artifacts.utils import mog_3d_tensor
from monai.transforms import Orientation
import nibabel as nib


class FetalSynthGen:

    def __init__(
        self,
        shape: Iterable[int],
        resolution: Iterable[float],
        device: str,
        intensity_generator: ImageFromSeeds,
        spatial_deform: SpatialDeformation,
        resampler: RandResample,
        bias_field: RandBiasField,
        noise: RandNoise,
        gamma: RandGamma,
        # optional SR artifacts
        blur_cortex: BlurCortex | None = None,
        struct_noise: StructNoise | None = None,
        simulate_motion: SimulateMotion | None = None,
        boundaries: SimulatedBoundaries | None = None,
        # optional
        stack_sampler: StackSampler | None = None,
    ):
        """
        Initialize the model with the given parameters.

        !!!Note
            Augmentations related to SR artifacts are optional and can be set to None
            if not needed.

        Args:
            shape: Shape of the output image.
            resolution: Resolution of the output image.
            device: Device to use for computation.
            intensity_generator: Intensity generator.
            spatial_deform: Spatial deformation generator.
            resampler: Resampler.
            bias_field: Bias field generator.
            noise: Noise generator.
            gamma: Gamma correction generator.
            blur_cortex: Cortex blurring generator.
            struct_noise: Structural noise generator.
            simulate_motion: Motion simulation generator.
            boundaries: Boundaries generator
            stack_sampler: Stack sampler.

        """
        self.shape = shape
        self.resolution = resolution
        self.intensity_generator = intensity_generator
        self.spatial_deform = spatial_deform
        self.resampled = resampler
        self.biasfield = bias_field
        self.gamma = gamma
        self.noise = noise

        self.artifacts = {
            "blur_cortex": blur_cortex,
            "struct_noise": struct_noise,
            "simulate_motion": simulate_motion,
            "boundaries": boundaries,
        }
        self.stack_sampler = stack_sampler
        self.device = device

    def _validated_genparams(self, d: dict) -> dict:
        """Recursively removes all the keys with None values as they are not fixed in the generation."""
        if not isinstance(d, dict):
            return d  # Return non-dictionaries as-is

        return {
            key: self._validated_genparams(value)
            for key, value in d.items()
            if value is not None
        }

    def sample(
        self,
        orientation,
        image: torch.Tensor | None,
        segmentation: torch.Tensor,
        seeds: torch.Tensor | None,
        genparams: dict = {},
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Generate a synthetic image from the input data.
        Supports both random generation and from a fixed genparams dictionary.

        Args:
            image: Image to use as intensity prior if required.
            segmentation: Segmentation to use as spatial prior.
            seeds: Seeds to use for intensity generation.
            genparams: Dictionary with generation parameters.
                Used for fixed generation.
                Should follow the structure and be of the same type as
                the returned generation parameters.

        Returns:
            The synthetic image, the segmentation, the original image, and the generation parameters.

        """
        if genparams:
            genparams = self._validated_genparams(genparams)

        # 1. Generate intensity output.
        if seeds is not None:
            seeds, selected_seeds = self.intensity_generator.load_seeds(
                seeds=seeds,
                genparams=genparams.get("selected_seeds", {}),
                orientation=orientation,
            )
            output, seed_intensities = self.intensity_generator.sample_intensities(
                seeds=seeds,
                device=self.device,
                genparams=genparams.get("seed_intensities", {}),
            )
        else:
            if image is None:
                raise ValueError(
                    "If no seeds are passed, an image must be loaded to be used as intensity prior!"
                )
            # normalize the image from 0 to 255 to
            # match the intensity generator
            output = (image - image.min()) / (image.max() - image.min()) * 255
            selected_seeds = {}
            seed_intensities = {}

        # ensure that tensors are on the same device
        output = output.to(self.device)
        segmentation = segmentation.to(self.device)
        image = image.to(self.device) if image is not None else None

        # 2. Spatially deform the data
        image, segmentation, output, deform_params = self.spatial_deform.deform(
            image=image,
            segmentation=segmentation,
            output=output,
            genparams=genparams.get("deform_params", {}),
        )

        # 3. Gamma contrast transformation
        output, gamma_params = self.gamma(
            output, self.device, genparams=genparams.get("gamma_params", {})
        )

        # 4. Bias field corruption
        output, bf_params = self.biasfield(
            output, self.device, genparams=genparams.get("bf_params", {})
        )

        # 5. Downsample to simulate lower reconstruction resolution
        output, factors, resample_params = self.resampled(
            output,
            np.array(self.resolution),
            self.device,
            genparams=genparams.get("resample_params", {}),
        )

        # 6. Noise corruption
        output, noise_params = self.noise(
            output, self.device, genparams=genparams.get("noise_params", {})
        )

        # 7. Up-sample back to the original resolution/shape
        output = self.resampled.resize_back(output, factors)

        # 8. Induce SR-artifacts
        artifacts = {}
        for name, artifact in self.artifacts.items():
            if artifact is not None:
                output, metadata = artifact(
                    output,
                    segmentation,
                    self.device,
                    genparams.get("artifact_params", {}),
                    resolution=self.resolution,
                )
                artifacts[name] = metadata

        # 8.1 Apply random translation
        if False:  # torch.rand(1).item() < 0.5:
            translation = RandomTranslation3D(min_voxel=-20, max_voxel=20)
            output = translation(output)
            if image is not None:
                image = translation(image)
            if segmentation is not None:
                segmentation = translation(segmentation)

            # 9. Apply stack sampler if available
            if self.stack_sampler is not None:
                output, segmentation, meta = self.stack_sampler(
                    output, segmentation, device=self.device
                )

                # unsqueeze the image to match the expected shape
                output = output.squeeze(0)
                segmentation = segmentation.squeeze(0)

        # 10. Aggregete the synth params
        synth_params = {
            "selected_seeds": selected_seeds,
            "seed_intensities": seed_intensities,
            "deform_params": deform_params,
            "gamma_params": gamma_params,
            "bf_params": bf_params,
            "resample_params": resample_params,
            "noise_params": noise_params,
            "artifacts": artifacts,
            "stack_sampler": meta if self.stack_sampler is not None else None,
        }

        return output, segmentation, image, synth_params


import torch
import random


class RandomTranslation3D:
    """
    Randomly translates a 3D image tensor.

    The class accepts minimum and maximum voxel values and randomly
    selects an integer shift for each spatial dimension (depth, height, width)
    between these limits. For 3D tensors, the translation is applied along the
    dimensions (0, 1, 2). For 4D tensors (e.g., with a channel dimension), the
    translation is applied along dimensions (1, 2, 3).

    Parameters:
        min_voxel (int): Minimum translation (in voxels). Can be negative.
        max_voxel (int): Maximum translation (in voxels).

    Example:
        translator = RandomTranslation3D(min_voxel=-5, max_voxel=5)
        translated_image = translator(image_tensor)
    """

    def __init__(self, min_voxel: int, max_voxel: int):
        if min_voxel > max_voxel:
            raise ValueError("min_voxel should not be greater than max_voxel")
        self.min_voxel = min_voxel
        self.max_voxel = max_voxel

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply a random translation to the given image tensor.

        Args:
            image (torch.Tensor): A 3D image tensor of shape (D, H, W) or a 4D tensor of shape (C, D, H, W).

        Returns:
            torch.Tensor: The randomly translated image.
        """
        # Generate random shifts for each spatial dimension.
        shift_d = random.randint(self.min_voxel, self.max_voxel)
        shift_h = random.randint(self.min_voxel, self.max_voxel)
        shift_w = random.randint(self.min_voxel, self.max_voxel)

        if image.dim() == 3:
            # For image tensor with shape [D, H, W]
            shifted_image = torch.roll(
                image, shifts=(shift_d, shift_h, shift_w), dims=(0, 1, 2)
            )
        elif image.dim() == 4:
            # For image tensor with shape [C, D, H, W]
            shifted_image = torch.roll(
                image, shifts=(shift_d, shift_h, shift_w), dims=(1, 2, 3)
            )
        else:
            raise ValueError("Input image tensor must be either 3D or 4D.")

        return shifted_image
