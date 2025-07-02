from typing import Iterable

import numpy as np

from torch import Tensor
from torchio.transforms import Compose, RescaleIntensity
from torchio import ScalarImage, LabelMap, Subject

from fetalsyngen.generator.intensity.rand_gmm import ImageFromSeeds
from fetalsyngen.generator.augmentation.skullstripping import SimulatedBoundaries


class FetalSynthGen:
    def __init__(
        self,
        shape: Iterable[int],
        resolution: Iterable[float],
        device: str,
        intensity_generator: ImageFromSeeds,
        image_transforms: Compose,
        boundaries_transform: SimulatedBoundaries | None = None,
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
            image_transforms: Image transforms to apply.

        """
        self.shape = shape
        self.resolution = resolution
        self.intensity_generator = intensity_generator
        self.image_transforms = image_transforms
        self.boundaries_transform = boundaries_transform
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
        image: Tensor | None,
        segmentation: LabelMap,
        seeds: dict | None,
        genparams: dict = {},
    ) -> tuple[Subject, dict]:
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
                subject: TorchIO Subject containing the generated image and segmentation.
                synth_params: Dictionary with the generation parameters used for this sample.
        """
        if genparams:
            genparams = self._validated_genparams(genparams)

        # 1. Generate intensity output
        if seeds is not None:
            seeds, selected_seeds = self.intensity_generator.load_seeds(
                seeds=seeds, genparams=genparams.get("selected_seeds", {})
            )
            # 1000 ms
            # 1.5 Simulate skull stripping boundaries
            if self.boundaries_transform is not None:
                # apply the boundaries transform to the seeds
                seeds, skullstrip_params = self.boundaries_transform(
                    seeds=seeds.data,
                )

            output, seed_intensities = self.intensity_generator.sample_intensities(
                seeds=seeds,
                device=self.device,
                genparams=genparams.get("seed_intensities", {}),
            )

            # turn output to torchio.ScalarImage
            output = ScalarImage(
                tensor=output,  # Add batch and channel dimensions
                affine=segmentation.affine,
            )
        else:
            if image is None:
                raise ValueError(
                    "If no seeds are passed, an image must be loaded to be used as intensity prior!"
                )
            # normalize the image from 0 to 255 to
            # match the intensity generator
            output = RescaleIntensity(out_min_max=(0, 255))(image)
            # selected_seeds = {}
            # seed_intensities = {}
            # skullstrip_params = {}

        # make output a torchio.ScalarImage
        # output = ScalarImage(
        #     tensor=output,  # Add batch and channel dimensions
        #     affine=segmentation.affine,
        # )

        # print(f"Type of output: {type(output)} segmentation: {type(segmentation)}")
        # combine the output and segmentation into a TorchIO subject to
        # ensure applied transformations are consistent
        subject = Subject(image=output, label=segmentation)

        # 2. Apply main torchio transforms
        subject = self.image_transforms(subject)

        output = subject["image"]
        segmentation = subject["label"]

        # torcio_params = {tr: trparm for tr, trparm in subject.applied_transforms}
        # aggregate all the parameters
        # synth_params = (
        #     torcio_params | selected_seeds | skullstrip_params | seed_intensities
        # )
        synth_params = {}
        return subject, synth_params


if __name__ == "__main__":

    from fetalsyngen.data.datasets import FetalSynthDataset
    import time
    import nibabel as nib
    from torchio.transforms import (
        RandomAffineElasticDeformation,
        RandomAnisotropy,
        Compose,
        RandomBiasField,
        RandomGamma,
        RandomNoise,
        RandomMotion,
        RandomGhosting,
        RandomSpike,
        RandomBlur,
        RandomFlip,
    )

    # intensity transforms: ~500ms

    # blur: 500ms
    blurtransf = RandomBlur(
        std=(0, 1),
        p=1,
    )

    # randaffineelastic: 23000ms
    spatial_deform = RandomAffineElasticDeformation(
        p=0.9,
        affine_first=False,
        affine_kwargs={
            "scales": 0.1,
            "degrees": 20,
            "translation": 5,
            "isotropic": False,
        },
        elastic_kwargs={
            "num_control_points": 7,
            "max_displacement": 8,
        },
    )

    # randflip: 50ms
    randflip = RandomFlip(
        axes=("L", "R"),
        flip_probability=0.5,
    )

    # 100ms
    resampletransf = RandomAnisotropy(
        axes=(0, 1, 2),
        downsampling=(1.5, 5),
        scalars_only=True,
        p=1.0,
    )

    # randgamma: 50ms
    gammatransf = RandomGamma(
        log_gamma=(-0.5, 0.5),
        p=0.8,
    )

    # biasfield: 50ms
    biasfield = RandomBiasField(coefficients=0.5, order=1, p=0.2)

    # MR artifacts: ~200ms
    randmotion = RandomMotion(
        degrees=10,
        translation=5,
        num_transforms=1,
        p=0.2,
    )

    # 300ms
    randghosting = RandomGhosting(
        num_ghosts=(1, 10),
        intensity=(0.1, 0.5),
        p=0.2,
    )

    # 500ms
    randspike = RandomSpike(
        num_spikes=(1, 3),
        intensity=0.3,
        p=0.2,
    )

    # 100ms
    noistransf = RandomNoise(
        mean=1.0,
        std=(0, 0.25),
        p=0.2,
    )

    generator = FetalSynthGen(
        shape=(256, 256, 256),
        resolution=(0.5, 0.5, 0.5),
        device="cpu",
        intensity_generator=ImageFromSeeds(
            min_subclusters=1,
            max_subclusters=3,
            seed_labels=list(range(100)),
            generation_classes=list(range(100)),
            meta_labels=4,
            empty_background=0.5,  # 50% chance to have empty background
        ),
        image_transforms=Compose(
            [
                # smooth synth data
                blurtransf,
                # spatial
                spatial_deform,
                randflip,
                resampletransf,
                # intensity
                gammatransf,
                biasfield,
                # # MR artifacts | AFTER MOTION AND SPATIAL DEFORMATIONS
                randmotion,
                randghosting,
                randspike,
                # last one so it's not cancelled by other transforms
                noistransf,
            ]
        ),
        boundaries_transform=SimulatedBoundaries(
            p=0.5,
            skullmetalabel=4,
            min_skull_erosion=0.02,
            max_kernel_size=8,  # size of the kernel to use for erosion/dilation
        ),
    )

    dataset = FetalSynthDataset(
        bids_path="/media/vzalevskyi/data/FETA_challenge/merged_feta_spinabifida/derivatives/resampled05",
        generator=generator,
        seed_path="/media/vzalevskyi/data/FETA_challenge/merged_feta_spinabifida/derivatives/seeds",
        sub_list=["sub-050"],
    )

    print(f"Dataset length: {len(dataset)}")
    for i in range(1):
        starttime = time.time()
        sample = dataset[i]
        print(f"Generated sample {i} in {time.time() - starttime:.2f} seconds")
        img = sample["image"]
        seg = sample["label"]

        nibimage = nib.Nifti1Image(
            img.numpy().astype(np.float32)[0],
            affine=seg.affine,
        )
        nib.save(nibimage, f"sample_{i}_image.nii.gz")
        nibseg = nib.Nifti1Image(
            seg.numpy().astype(np.int16)[0],
            affine=seg.affine,
        )
        nib.save(nibseg, f"sample_{i}_label.nii.gz")
