import torch
from pathlib import Path
import numpy as np
from fetalsyngen.utils.image_reading import SimpleITKReader
from typing import Iterable
from monai.transforms import Orientation


class ImageFromSeeds:

    def __init__(
        self,
        min_subclusters: int,
        max_subclusters: int,
        seed_labels: Iterable[int],
        generation_classes: Iterable[int],
        meta_labels: int = 4,
    ):
        """

        Args:
            min_subclusters (int): Minimum number of subclusters to use.
            max_subclusters (int): Maximum number of subclusters to use.
            seed_labels (Iterable[int]): Iterable with all possible labels
                that can occur in the loaded seeds. Should be a unique set of
                integers starting from [0, ...]. 0 is reserved for the background,
                that will not have any intensity generated.
            generation_classes (Iterable[int]): Classes to use for generation.
                Seeds with the same generation calss will be generated with
                the same GMM. Should be the same length as seed_labels.
            meta_labels (int, optional): Number of meta-labels used. Defaults to 4.
        """
        self.min_subclusters = min_subclusters
        self.max_subclusters = max_subclusters
        try:
            assert len(set(seed_labels)) == len(seed_labels)
        except AssertionError:
            raise ValueError("Parameter seed_labels should have unique values.")
        try:
            assert len(seed_labels) == len(generation_classes)
        except AssertionError:
            raise ValueError(
                "Parameters seed_labels and generation_classes should have the same lengths."
            )
        self.seed_labels = seed_labels
        self.generation_classes = generation_classes
        self.meta_labels = meta_labels
        self.loader = SimpleITKReader()
        self.orientation = Orientation(axcodes="RAS")

    def load_seeds(
        self,
        seeds: dict[int : dict[int:Path]],
        mlabel2subclusters: dict[int:int] | None = None,
        genparams: dict = {},
    ) -> torch.Tensor:
        """Generate an intensity image from seeds.
        If seed_mapping is provided, it is used to
        select the number of subclusters to use for
        each meta label. Otherwise, the number of subclusters
        is randomly selected from a uniform discrete distribution
        between `min_subclusters` and `max_subclusters` (both inclusive).

        Args:

            seeds: Dictionary with the mapping `subcluster_number: {meta_label: seed_path}`.
            mlabel2subclusters: Mapping to use when defining how many subclusters to
                use for each meta-label. Defaults to None.
            genparams: Dictionary with generation parameters. Defaults to {}.
                Should contain the key "mlabel2subclusters" if the mapping is to be fixed.


        Returns:
            torch.Tensor: Intensity image with the same shape as the seeds.
                Tensor dimensions are **(H, W, D)**. Values inside the tensor
                correspond to the subclusters, and are grouped by meta-label.
                `1-19: CSF, 20-29: GM, 30-39: WM, 40-49: Extra-cerebral`.
        """
        # if no mapping is provided, randomly select the number of subclusters
        # to use for each meta-label in the format {mlabel: n_subclusters}
        if mlabel2subclusters is None:
            mlabel2subclusters = {
                meta_label: np.random.randint(
                    self.min_subclusters, self.max_subclusters + 1
                )
                for meta_label in range(1, self.meta_labels + 1)
            }
        if "mlabel2subclusters" in genparams.keys():
            mlabel2subclusters = genparams["mlabel2subclusters"]

        # load the first seed as the one corresponding to mlabel 1
        seed = self.loader(seeds[mlabel2subclusters[1]][1])
        seed = self.orientation(seed.unsqueeze(0))
        # re-orient seeds to RAS

        for mlabel in range(2, self.meta_labels + 1):
            new_seed = self.loader(seeds[mlabel2subclusters[mlabel]][mlabel])
            new_seed = self.orientation(new_seed.unsqueeze(0))
            seed += new_seed

        return seed.long().squeeze(0), {"mlabel2subclusters": mlabel2subclusters}

    def sample_intensities(
        self, seeds: torch.Tensor, device: str, genparams: dict = {}
    ) -> torch.Tensor:
        """Sample the intensities from the seeds.

        Args:
            seeds (torch.Tensor): Tensor with the seeds.
            device (str): Device to use. Should be "cuda" or "cpu".
            genparams (dict, optional): Dictionary with generation parameters.
                Defaults to {}. Should contain the keys "mus" and "sigmas" if
                the GMM parameters are to be fixed.

        Returns:
            torch.Tensor: Tensor with the intensities.
        """
        nlabels = max(self.seed_labels) + 1
        nsamp = len(self.seed_labels)

        # # Sample GMMs means and stds
        mus = (
            25 + 200 * torch.rand(nlabels, dtype=torch.float, device=device)
            if "mus" not in genparams.keys()
            else genparams["mus"]
        )
        sigmas = (
            5
            + 20
            * torch.rand(
                nlabels,
                dtype=torch.float,
                device=device,
            )
            if "sigmas" not in genparams.keys()
            else genparams["sigmas"]
        )

        # if there are seed labels from the same generation class
        # set their mean to be the same with some random perturbation
        if self.generation_classes != self.seed_labels:
            mus[self.seed_labels] = torch.clamp(
                mus[self.generation_classes]
                + 25 * torch.randn(nsamp, dtype=torch.float, device=device),
                0,
                225,
            )
        intensity_image = mus[seeds] + sigmas[seeds] * torch.randn(
            seeds.shape, dtype=torch.float, device=device
        )
        intensity_image[intensity_image < 0] = 0

        return intensity_image, {
            "mus": mus,
            "sigmas": sigmas,
        }
