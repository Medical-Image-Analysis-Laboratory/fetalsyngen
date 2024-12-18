import torch
import monai
import numpy as np
from monai.transforms import Spacing
from fetalsyngen.utils.brainid import (
    gaussian_blur_3d,
    fast_3D_interp_torch,
    myzoom_torch,
)

# TODO: Add device tracking, invertability and parameter logging through decorators
# as well as random parameter selection
# TODO: descibe inputs outputs and docs for classes


class RandTransform(monai.transforms.Transform):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def random_uniform(self, min_val, max_val):
        return np.random.uniform(min_val, max_val)


class RandResample(RandTransform):
    def __init__(
        self,
        prob: float,
        min_resolution: float,
        max_resolution: float,
    ):
        self.prob = prob
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution

    def __call__(self, output, input_resolution, device):
        if np.random.rand() < self.prob:
            input_size = np.array(output.shape)
            spacing = np.array([1.0, 1.0, 1.0]) * self.random_uniform(
                self.min_resolution, self.max_resolution
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

            output_reized = fast_3D_interp_torch(output_blurred, II, JJ, KK, "linear")
            return output_reized, factors, {"spacing": spacing}
        else:
            return output, None, {"spacing": None}

    def resize_back(self, output_reized, factors):
        if factors is not None:
            output_reized = myzoom_torch(output_reized, 1 / factors)
            return output_reized / torch.max(output_reized)
        else:
            return output_reized


class RandBiasField(RandTransform):

    def __init__(self, prob, scale_min, scale_max, std_min, std_max):
        self.prob = prob
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.std_min = std_min
        self.std_max = std_max

    def __call__(self, output, device):
        image_size = output.shape
        bf_scale = self.scale_min + np.random.rand(1) * (
            self.scale_max - self.scale_min
        )
        bf_size = np.round(bf_scale * np.array(image_size)).astype(int).tolist()
        bf_std = self.std_min + (self.std_max - self.std_min) * np.random.rand(1)

        bf_low_scale = torch.tensor(
            bf_std,
            dtype=torch.float,
            device=device,
        ) * torch.randn(bf_size, dtype=torch.float, device=device)
        bf_interp = myzoom_torch(bf_low_scale, np.array(image_size) / bf_size)
        bf = torch.exp(bf_interp)

        return output * bf, {"bf_scale": bf_scale, "bf_std": bf_std, "bf_size": bf_size}


class RandNoise(RandTransform):
    def __init__(self, prob, std_min, std_max):
        self.prob = prob
        self.std_min = std_min
        self.std_max = std_max

    def __call__(self, output, device):
        noise_std = self.std_min + (self.std_max - self.std_min) * np.random.rand(1)

        noise_std = torch.tensor(
            noise_std,
            dtype=torch.float,
            device=device,
        )
        SYN_noisy = output + noise_std * torch.randn(
            output.shape, dtype=torch.float, device=device
        )
        SYN_noisy[SYN_noisy < 0] = 0
        return output, {"noise_std": noise_std}


class RandGamma(RandTransform):
    def __init__(self, prob, gamma_std):
        self.prob = prob
        self.gamma_std = gamma_std

    def __call__(self, output, device):
        gamma = None
        if np.random.rand() < self.prob:
            gamma = np.exp(self.gamma_std * np.random.randn(1)[0])
            gamma_tensor = torch.tensor(
                gamma,
                dtype=float,
                device=device,
            )
            output = 300.0 * (output / 300.0) ** gamma_tensor
        return output, {"gamma": gamma}
