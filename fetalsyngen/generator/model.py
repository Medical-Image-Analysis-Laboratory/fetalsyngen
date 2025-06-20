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
)
from fetalsyngen.generator.artifacts.utils import mog_3d_tensor


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
        self.device = device

    def _validated_genparams(self, d: dict) -> dict:
        """Recursively removes all the keys with None values as they are not fixed in the generation."""
        if not isinstance(d, dict):
            return d  # Return non-dictionaries as-is

        return {
            key: self._validated_genparams(value) for key, value in d.items() if value is not None
        }

    def generate(
        self,
        image: torch.Tensor | None,
        segmentation: torch.Tensor,
        seeds: torch.Tensor | None,
        genparams: dict = {},
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Generate a synthetic deformed image from the input data.
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
            The synthetic deformed image, the deformed segmentation, the original image, and the generation parameters.

        """

        # 1. Generate intensity output.
        if seeds is not None:
            seeds, selected_seeds = self.intensity_generator.load_seeds(
                seeds=seeds, genparams=genparams.get("selected_seeds", {})
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
        synth_params = {
            "selected_seeds": selected_seeds,
            "seed_intensities": seed_intensities,
            "deform_params": deform_params,
        }
        return output, segmentation, image, synth_params

    def augment(
        self,
        image: torch.Tensor | None,
        segmentation: torch.Tensor,
        genparams: dict = {},
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Generate a randomized augmentations on the synthetic image from the input data.
        Supports both random generation and from a fixed genparams dictionary.

        Args:
            image: Image to be augmented
            segmentation: Segmentation to use as spatial prior.
            genparams: Dictionary with generation parameters.
                Used for fixed generation.
                Should follow the structure and be of the same type as
                the returned generation parameters.

        Returns:
            The synthetic augmented image, the segmentation and the generation parameters.

        """
        # 1. Gamma contrast transformation
        output, gamma_params = self.gamma(
            image, self.device, genparams=genparams.get("gamma_params", {})
        )

        # 2. Bias field corruption
        output, bf_params = self.biasfield(
            output, self.device, genparams=genparams.get("bf_params", {})
        )

        # 3. Downsample to simulate lower reconstruction resolution
        output, factors, resample_params = self.resampled(
            output,
            np.array(self.resolution),
            self.device,
            genparams=genparams.get("resample_params", {}),
        )

        # 4. Noise corruption
        output, noise_params = self.noise(
            output, self.device, genparams=genparams.get("noise_params", {})
        )

        # 5. Up-sample back to the original resolution/shape
        output = self.resampled.resize_back(output, factors)

        # 6. Induce SR-artifacts
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

        synth_params = {
            "gamma_params": gamma_params,
            "bf_params": bf_params,
            "resample_params": resample_params,
            "noise_params": noise_params,
            "artifacts": artifacts,
        }
        return output, synth_params

    def sample(
        self,
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

        # 1. Generate the deformed image
        output, segmentation, image, synth_params = self.generate(
            image=image,
            segmentation=segmentation,
            seeds=seeds,
            genparams=genparams,
        )

        # 2. Augment the deformed image
        output, synth_params_aug = self.augment(
            image=output,
            segmentation=segmentation,
            genparams=genparams,
        )

        # 3. Aggregete the synth params
        synth_params.update(synth_params_aug)

        return output, segmentation, image, synth_params
