import numpy as np
import torch
from .synthseg import RandTransform
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    generate_binary_structure,
    iterate_structure,
)


class SimulatedBoundaries(RandTransform):
    """
    Simulates various types of boundaries in the image, either doing no masking
    (with probability `prob_no_mask`), adding a halo around the mask (with probability
    `prob_if_mask_halo`), or adding fuzzy boundaries to the mask (with probability `prob_if_mask_fuzzy`).
    """

    def __init__(
        self,
        p: float = 0.5,
        skullmetalabel: int = 4,
        min_skull_erosion: float = 0.02,
        max_kernel_size: int = 8,
    ):
        """
        Initialize the augmentation parameters.

        Args:
            p (float): Probability of applying the augmentation.
                Defaults to 0.5.
            skullmetalabel (int): Label of the skull (extra-cerebral tissues)
                meta-label to use for augmentation. Means that all all labels
                of the seeds of >skullmetalabel*10 will be considered as skull
                Defaults to 4. NOTE: the skullmetalabel should be the last one.
            min_skull_erosion (float): Minimum fraction of the skull to erode.
                Defaults to 0.02. This means that if the skull is least 2%
                of the whole volume, it will be eroded.
            max_kernel_size (int): Size of the kernel to use for erosion/dilation.

        """
        self.p = p
        self.skullmetalabel = skullmetalabel
        self.min_skull_erosion = min_skull_erosion
        self.max_kernel_size = max_kernel_size
        self.kernel_size = np.random.randint(1, self.max_kernel_size + 1)
        self.kernel = generate_binary_structure(
            3,  # 3D structure
            1,  # connectivity
        )
        self.kernel = iterate_structure(self.kernel, self.kernel_size)

    def __call__(self, seeds) -> tuple[torch.Tensor, dict]:
        if np.random.rand() > self.p:
            # do not apply the augmentation
            return seeds, {}

        brain_skull_mask = seeds > 0
        brain_mask = (seeds <= (self.skullmetalabel * 10)) & (seeds > 0)
        skull_mask = seeds > (self.skullmetalabel * 10)

        skull_proportion = skull_mask.sum() / (
            seeds.shape[0] * seeds.shape[1] * seeds.shape[2]
        )

        if skull_proportion > self.min_skull_erosion:
            # only erode
            brain_skull_mask = binary_erosion(
                brain_skull_mask[0], structure=self.kernel
            )
        else:
            # else apply randomly either erosion or dilation
            if np.random.rand() < 0.5:
                # erode the skull mask
                brain_skull_mask = binary_erosion(
                    brain_skull_mask[0], structure=self.kernel
                )
            else:
                # dilate the skull mask
                brain_skull_mask = binary_dilation(
                    brain_skull_mask[0], structure=self.kernel
                )
        brain_skull_mask = brain_skull_mask[None, :, :, :].astype(bool)
        brain_skull_mask = torch.from_numpy(brain_skull_mask)
        seeds[(brain_skull_mask) & (~brain_mask) & (~skull_mask)] = (
            self.skullmetalabel * 10 + 3
        )

        return seeds, {
            "skull_proportion": skull_proportion,
            "max_kernel_size": self.max_kernel_size,
            "min_skull_erosion": self.min_skull_erosion,
            "kernel_size": self.kernel_size,
        }
