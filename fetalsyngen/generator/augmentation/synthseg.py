import torch
import monai
import numpy as np
from monai.transforms import Spacing
from fetalsyngen.utils.generation import (
    gaussian_blur_3d,
    fast_3D_interp_torch,
    myzoom_torch,
)

# TODO: Thomas add deterministic augmentation option


class RandTransform(monai.transforms.Transform):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def random_uniform(self, min_val, max_val):
        return np.random.uniform(min_val, max_val)


class RandResample(RandTransform):
    """Resample the input image to a random resolution sampled uniformly between
    `min_resolution` and `max_resolution` with a probability of `prob`.

    If the resolution is smaller than the input resolution, no resampling is performed.
    """

    def __init__(
        self,
        prob: float,
        min_resolution: float,
        max_resolution: float,
    ):
        """
        Initialize the augmentation parameters.

        Args:
            prob (float): Probability of applying the augmentation.
            min_resolution (float): Minimum resolution for the augmentation (in mm).
            max_resolution (float): Maximum resolution for the augmentation.
        """
        self.prob = prob
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution

    def __call__(self, output, input_resolution, device, genparams: dict = {}):
        if np.random.rand() < self.prob or "spacing" in genparams.keys():
            input_size = np.array(output.shape)
            spacing = (
                np.array([1.0, 1.0, 1.0])
                * self.random_uniform(self.min_resolution, self.max_resolution)
                if "spacing" not in genparams.keys()
                else genparams["spacing"]
            )

            # calculate stds of gaussian kernels
            # used for blurring to simulate resampling
            # the data to different resolutions
            stds = (
                (0.85 + 0.3 * np.random.rand())
                * np.log(5)
                / np.pi
                * spacing
                / input_resolution
            )
            # no blur if thickness is equal or smaller to the resolution of the training data
            stds[spacing <= input_resolution] = 0.0
            output_blurred = gaussian_blur_3d(output, stds, device)

            # resize the blurred output to the new resolution
            new_size = (np.array(input_size) * input_resolution / spacing).astype(int)

            # calculate the factors for the interpolation
            factors = np.array(new_size) / np.array(input_size)
            # delta is the offset for the interpolation
            delta = (1.0 - factors) / (2.0 * factors)
            vx = np.arange(
                delta[0], delta[0] + new_size[0] / factors[0], 1 / factors[0]
            )[: new_size[0]]
            vy = np.arange(
                delta[1], delta[1] + new_size[1] / factors[1], 1 / factors[1]
            )[: new_size[1]]
            vz = np.arange(
                delta[2], delta[2] + new_size[2] / factors[2], 1 / factors[2]
            )[: new_size[2]]
            II, JJ, KK = np.meshgrid(vx, vy, vz, sparse=False, indexing="ij")
            II = torch.tensor(II, dtype=torch.float, device=device)
            JJ = torch.tensor(JJ, dtype=torch.float, device=device)
            KK = torch.tensor(KK, dtype=torch.float, device=device)

            output_resized = fast_3D_interp_torch(output_blurred, II, JJ, KK, "linear")
            return output_resized, factors, {"spacing": spacing}
        else:
            return output, None, {"spacing": None}

    def resize_back(self, output_resized, factors):
        if factors is not None:
            output_resized = myzoom_torch(output_resized, 1 / factors)
            return output_resized / torch.max(output_resized)
        else:
            return output_resized


class RandBiasField(RandTransform):
    """Add a random bias field to the input image with a probability of `prob`."""

    def __init__(
        self,
        prob: float,
        scale_min: float,
        scale_max: float,
        std_min: float,
        std_max: float,
    ):
        """

        Args:
            prob: Probability of applying the augmentation.
            scale_min: Minimum scale of the bias field.
            scale_max: Maximum scale of the bias field.
            std_min: Minimum standard deviation of the bias field.
            std_max: Maximum standard deviation of the bias.
        """

        self.prob = prob
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.std_min = std_min
        self.std_max = std_max

    def __call__(self, output, device, genparams: dict = {}):
        if np.random.rand() < self.prob or len(genparams.keys()) > 0:
            image_size = output.shape
            bf_scale = (
                self.scale_min + np.random.rand(1) * (self.scale_max - self.scale_min)
                if "bf_scale" not in genparams.keys()
                else genparams["bf_scale"]
            )
            bf_size = np.round(bf_scale * np.array(image_size)).astype(int).tolist()
            bf_std = (
                self.std_min + (self.std_max - self.std_min) * np.random.rand(1)
                if "bf_std" not in genparams.keys()
                else genparams["bf_std"]
            )

            bf_low_scale = torch.tensor(
                bf_std,
                dtype=torch.float,
                device=device,
            ) * torch.randn(bf_size, dtype=torch.float, device=device)
            bf_interp = myzoom_torch(bf_low_scale, np.array(image_size) / bf_size)
            bf = torch.exp(bf_interp)

            return output * bf, {
                "bf_scale": bf_scale,
                "bf_std": bf_std,
                "bf_size": bf_size,
            }
        else:
            return output, {"bf_scale": None, "bf_std": None, "bf_size": None}


class RandNoise(RandTransform):
    """Add random Gaussian noise to the input image with a probability of `prob`."""

    def __init__(self, prob: float, std_min: float, std_max: float):
        """
        The image scale is 0-255 so the noise is added in the same scale.
        Args:
            prob: Probability of applying the augmentation.
            std_min: Minimum standard deviation of the noise.
            std_max: Maximum standard deviation of the noise
        """
        self.prob = prob
        self.std_min = std_min
        self.std_max = std_max

    def __call__(self, output, device, genparams: dict = {}):
        noise_std = None
        if np.random.rand() < self.prob or "noise_std" in genparams.keys():
            noise_std = (
                self.std_min + (self.std_max - self.std_min) * np.random.rand(1)
                if "noise_std" not in genparams.keys()
                else genparams["noise_std"]
            )

            noise_std = torch.tensor(
                noise_std,
                dtype=torch.float,
                device=device,
            )
            output = output + noise_std * torch.randn(
                output.shape, dtype=torch.float, device=device
            )
            output[output < 0] = 0
        return output, {"noise_std": noise_std}


class RandGamma(RandTransform):
    """Apply gamma correction to the input image with a probability of `prob`."""

    def __init__(self, prob: float, gamma_std: float):
        """
        Args:
            prob: Probability of applying the augmentation.
            gamma_std: Standard deviation of the gamma correction.
        """
        self.prob = prob
        self.gamma_std = gamma_std

    def __call__(self, output, device, genparams: dict = {}):
        gamma = None
        if np.random.rand() < self.prob or "gamma" in genparams.keys():
            gamma = (
                np.exp(self.gamma_std * np.random.randn(1)[0])
                if "gamma" not in genparams.keys()
                else genparams["gamma"]
            )
            gamma_tensor = torch.tensor(
                gamma,
                dtype=float,
                device=device,
            )
            output = 300.0 * (output / 300.0) ** gamma_tensor
        return output, {"gamma": gamma}
