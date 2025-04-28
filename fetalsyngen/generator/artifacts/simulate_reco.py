"""
This file contains the classes that are used to cpplate randomized acquisitions and reconstructions, as well as
boundary modifications.

- PSFReconstruction: Basic class that reconstructs the volume from the acquired slices using the PDF and the transforms given.
- Scanner: Class that simulates the acquisition of slices from a volume. Modified version from SVoRT (Xu et al. 2022.)
    at https://github.com/daviddmc/SVoRT/blob/main/src/data/scan.py
- PSFReconstructor: Class that reconstructs the volume from the acquired slices using the PDF and the transforms given.
    Randomly applies: misregistration of a part of the slices, removal of a portion of the slices, merging of the volume with the ground truth according to a 3D
    mixture of Gaussians and smoothing of the volume.
- SimulatedBoundaries: Class that simulates the modification of the boundaries of the volume.
    Randomly applies: No masking at all, halo (like SVRTK), fuzzy boundaries.
"""

import numpy as np
import torch
import torch.nn.functional as F
from fetalsyngen.generator.artifacts.svort import (
    RigidTransform,
    mat_update_resolution,
    random_init_stack_transforms,
    reset_transform,
    slice_acquisition,
    slice_acquisition_adjoint,
    sample_motion,
    interleave_index,
    get_PSF,
    random_angle,
)
from functools import partial
from fetalsyngen.generator.artifacts.utils import mog_3d_tensor


def PSFreconstruction(transforms, slices, slices_mask, vol_mask, params):
    """
    Reconstruct the volume from the acquired slices using the PSF and the given transforms
    by calling the `slice_acquisition_adjoint` cuda method.

    """
    return slice_acquisition_adjoint(
        transforms,
        params["psf"],
        slices,
        slices_mask,
        vol_mask,
        params["volume_shape"],
        params["res_s"] / params["res_r"],
        params["interp_psf"],
        True,
    )


class Scanner:
    """
    Class that simulates the acquisition of slices from a volume.

    Samples multiple stacks of slices at various resolutions, various slice_thicnkesses and gaps.
    Adds artifacts to the simulated slices (noise, Gamma transform, signal voids) and spatial shifts.

    """

    def __init__(
        self,
        resolution_slice_fac_min,
        resolution_slice_fac_max,
        resolution_slice_max,
        slice_thickness_min,
        slice_thickness_max,
        gap_min,
        gap_max,
        min_num_stack,
        max_num_stack,
        max_num_slices,
        noise_sigma_min,
        noise_sigma_max,
        TR_min,
        TR_max,
        prob_gamma,
        gamma_std,
        prob_void,
        slice_size,
        restrict_transform: bool,
        txy: float,
        resolution_recon: float = None,
        slice_noise_threshold: float = 0.1,
    ):
        """
        Initialize the scanner with the given parameters.

        Args:
            resolution_slice_fac_min: Minimum slice resolution factor.
            resolution_slice_fac_max: Maximum slice resolution factor.
            resolution_slice_max: Maximum slice resolution.
            slice_thickness_min: Minimum slice thickness.
            slice_thickness_max: Maximum slice thickness.
            gap_min: Minimum gap between slices.
            gap_max: Maximum gap between slices.
            min_num_stack: Minimum number of stacks.
            max_num_stack: Maximum number of stacks.
            max_num_slices: Maximum number of slices.
            noise_sigma_min: Minimum noise sigma.
            noise_sigma_max: Maximum noise sigma.
            TR_min: Minimum TR.
            TR_max: Maximum TR.
            prob_gamma: Probability of applying the Gamma transform.
            gamma_std: Standard deviation of the Gamma transform.
            prob_void: Probability of applying the signal void.
            slice_size: Size of the slices.
            resolution_recon: Resolution of the reconstructed volume.
            restrict_transform: Restrict the transformation.
            txy: Translation factor.
            slice_noise_threshold: Slice noise threshold.

        """
        self.resolution_slice_fac_min = resolution_slice_fac_min
        self.resolution_slice_fac_max = resolution_slice_fac_max
        self.resolution_slice_max = resolution_slice_max
        self.slice_thickness_min = slice_thickness_min
        self.slice_thickness_max = slice_thickness_max
        self.gap_min = gap_min
        self.gap_max = gap_max
        self.min_num_stack = min_num_stack
        self.max_num_stack = max_num_stack
        self.max_num_slices = max_num_slices
        self.noise_sigma_min = noise_sigma_min
        self.noise_sigma_max = noise_sigma_max
        self.TR_min = TR_min
        self.TR_max = TR_max
        self.prob_gamma = prob_gamma
        self.gamma_std = gamma_std
        self.prob_void = prob_void
        self.slice_size = slice_size
        self.resolution_recon = resolution_recon
        self.restrict_transform = restrict_transform
        self.txy = txy
        self.slice_noise_threshold = slice_noise_threshold

    def get_resolution(self, data, genparams: dict = {}):
        """
        Resolution setting.
        - If resolution_slice_fac exists and is a list, then
            the slice resolution is uniformly sampled between
            resolution_slice_fac[0] and ...[1]*data["resolution"]
        - Else: resolution_slice = resolution_slice_fac*resolution
            if resolution_recon is set, it will be used.
            Otherwise,the resolution recon is sampled between
            data["resolution"] and resolution_slice.

        Then, the slice_thickness is either randomly sampled, same for gap.

        Args:
            data: Dictionary containing the data.
            genparams: Dictionary containing the generation parameters.

        Returns:
            dict: The updated data dictionary.
        """
        resolution = data["resolution"]
        if "resolution_slice_fac" not in genparams:
            resolution_slice = np.random.uniform(
                self.resolution_slice_fac_min * resolution,
                min(
                    self.resolution_slice_fac_max * resolution,
                    self.resolution_slice_max,
                ),
            )
        else:
            resolution_slice = genparams["resolution_slice_fac"]

        if self.resolution_recon is not None:
            data["resolution_recon"] = self.resolution_recon
        else:
            data["resolution_recon"] = np.random.uniform(
                resolution, resolution_slice
            )
        data["resolution_slice"] = resolution_slice
        if "slice_thickness" not in genparams:
            data["slice_thickness"] = np.random.uniform(
                self.slice_thickness_min, self.slice_thickness_max
            )
        else:
            data["slice_thickness"] = genparams["slice_thickness"]

        if "gap" not in genparams:
            data["gap"] = np.random.uniform(self.gap_min, self.gap_max)
        else:
            data["gap"] = genparams["gap"]

        return data

    def sample_time(self, n_slice, genparams: dict = {}):
        """
        Sample the time points for the slices.

        Args:
            n_slice: Number of slices.
            genparams: Dictionary containing the generation parameters.

        Returns:
            np.ndarray: The time points for the slices.
        """
        if "TR" not in genparams:
            TR = np.random.uniform(self.TR_min, self.TR_max)
        else:
            TR = genparams["TR"]
        return np.arange(n_slice) * TR

    def random_gamma(self, slices, genparams: dict = {}):
        """
        Apply the Gamma transform to the slices.

        Args:
            slices: The slices to apply the transform to.
            genparams: Dictionary containing the generation parameters.

        Returns:
            torch.Tensor: The transformed slices.
        """
        if np.random.rand() < self.prob_gamma:
            if "gamma" not in genparams:
                gamma = np.exp(self.gamma_std * np.random.randn(1)[0])
            else:
                gamma = genparams["gamma"]

            gamma = torch.tensor(
                gamma,
                dtype=float,
                device=slices.device,
            )
            slices = 300.0 * (slices / 300.0) ** gamma
            return slices / slices.max()
        return slices

    def add_noise(self, slices, genparams: dict = {}):
        """
        Add noise to the slices.

        Args:
            slices: The slices to add noise to.
            genparams: Dictionary containing the generation parameters.

        Returns:
            torch.Tensor: The noisy slices.
        """
        mask = slices > self.slice_noise_threshold
        masked = slices[mask]
        if "noise_sigma" not in genparams:
            sigma = np.random.uniform(
                self.noise_sigma_min, self.noise_sigma_max
            )
        else:
            sigma = genparams["noise_sigma"]
        noise1 = torch.randn_like(masked) * sigma
        noise2 = torch.randn_like(masked) * sigma
        slices[mask] = torch.sqrt((masked + noise1) ** 2 + noise2**2)
        return slices

    def signal_void(self, slices):
        """
        Apply signal voids to the slices.

        Args:
            slices: The slices to apply the signal voids to.

        Returns:
            torch.Tensor: The slices with the signal voids.
        """

        idx = (
            torch.rand(slices.shape[0], device=slices.device) < self.prob_void
        )

        n = idx.sum()
        if n > 0:
            h, w = slices.shape[-2:]
            y = torch.linspace(
                -(h - 1) / 2, (h - 1) / 2, h, device=slices.device
            )
            x = torch.linspace(
                -(w - 1) / 2, (w - 1) / 2, w, device=slices.device
            )
            yc = (torch.rand(n, device=slices.device) - 0.5) * (h - 1)
            xc = (torch.rand(n, device=slices.device) - 0.5) * (w - 1)

            y = y.view(1, -1, 1) - yc.view(-1, 1, 1)
            x = x.view(1, 1, -1) - xc.view(-1, 1, 1)

            theta = 2 * np.pi * torch.rand((n, 1, 1), device=slices.device)
            c = torch.cos(theta)
            s = torch.sin(theta)
            x, y = c * x - s * y, s * x + c * y

            # Re-tuning the signal drop parameters to enable larger
            # Signal drops on the images.
            a = 30 + torch.rand_like(theta) * 90
            A = torch.rand_like(theta) * 0.5 + 0.5
            sx = torch.rand_like(theta) * 30 + 39

            sy = a**2 / sx
            sx = -0.5 / sx**2
            sy = -0.5 / sy**2
            mask = 1 - A * torch.exp(sx * x**2 + sy * y**2)
            slices[idx, 0] *= mask
        return slices

    def scan(self, data, genparams: dict = {}):
        """
        Simulate the acquisition of slices from the volume.

        Args:
            data: Dictionary containing the data.
            genparams: Dictionary containing the generation parameters.

        Returns:
            dict: The updated data dictionary with simulated acquired slices.
        """
        data = self.get_resolution(data, genparams={})
        res = data["resolution"]
        res_r = data["resolution_recon"]
        res_s = data["resolution_slice"]
        s_thick = data["slice_thickness"]
        gap = data["gap"]
        device = data["volume"].device

        ## resample the ground truth if needed.
        if res_r != res:
            grids = []
            for i in range(3):
                size_new = int(data["volume"].shape[i + 2] * res / res_r)
                grid_max = (
                    (size_new - 1)
                    * res_r
                    / (data["volume"].shape[i + 2] - 1)
                    / res
                )
                grids.append(
                    torch.linspace(
                        -grid_max, grid_max, size_new, device=device
                    )
                )
            grid = torch.stack(
                torch.meshgrid(*grids, indexing="ij")[::-1], -1
            ).unsqueeze_(0)
            volume_gt = F.grid_sample(data["volume"], grid, align_corners=True)
            seg_gt = F.grid_sample(
                data["seg"], grid, mode="nearest", align_corners=True
            )
        else:
            volume_gt = data["volume"].clone()
            seg_gt = data["seg"].clone()
        data["volume_gt"] = volume_gt
        data["seg_gt"] = seg_gt

        # Define the PSF for the acquisition and the reconstruction.
        # They can be different because the original data can be at a higher resolution
        # than the target resolution.
        psf_acq = get_PSF(
            res_ratio=(res_s / res, res_s / res, s_thick / res), device=device
        )
        psf_rec = get_PSF(
            res_ratio=(res_s / res_r, res_s / res_r, s_thick / res_r),
            device=device,
        )
        data["psf_rec"] = psf_rec
        data["psf_acq"] = psf_acq

        # Voxel size? Not sure what this defines.
        vs = data["volume"].shape

        if self.slice_size is None:
            ss = int(
                np.sqrt((vs[-1] ** 2 + vs[-2] ** 2 + vs[-3] ** 2) / 2.0)
                * res
                / res_s
            )
            ss = int(np.ceil(ss / 32.0) * 32)
        else:
            ss = self.slice_size
        ns = int(max(vs) * res / gap) + 2

        ## Define the stacks and do the transformations.
        stacks = []
        stacks_no_psf = []
        transforms = []
        transforms_gt = []
        positions = []

        num_stacks = np.random.randint(
            self.min_num_stack, self.max_num_stack + 1
        )

        rand_motion = True
        while True:
            # print(f"Generating stack : {len(stacks)}")
            # stack transformation
            transform_init = random_init_stack_transforms(
                ns, gap, self.restrict_transform, self.txy, device
            )

            ts = self.sample_time(ns)

            transform_motion = sample_motion(ts, device, rand_motion)
            # interleaved acquisition
            interleave_idx = interleave_index(
                ns,
                (
                    np.random.randint(2, int(np.sqrt(ns)) + 1)
                    if rand_motion
                    else 2
                ),
            )
            transform_motion = transform_motion[interleave_idx]
            # apply motion
            transform_target = transform_motion.compose(transform_init)

            # sample slices
            mat = mat_update_resolution(transform_target.matrix(), res_r, res)
            slices = slice_acquisition(
                mat,
                data["volume"],
                None,
                None,
                psf_acq,
                (ss, ss),
                res_s / res,
                False,
                False,
            )
            slices_no_psf = slice_acquisition(
                mat,
                data["mask"],
                None,
                None,
                get_PSF(0, device=device),
                (ss, ss),
                res_s / res,
                False,
                False,
            )
            # remove zeros
            nnz = slices_no_psf.sum((1, 2, 3))
            idx = nnz > (nnz.max() * np.random.uniform(0.1, 0.3))
            if idx.sum() == 0:
                continue
            else:
                nz = torch.nonzero(idx)
                idx[nz[0, 0] : nz[-1, 0]] = True
            slices = slices[idx]
            slices_no_psf = slices_no_psf[idx]
            transform_init = transform_init[idx]
            transform_init = reset_transform(transform_init)
            transform_target = transform_target[idx]
            # artifacts
            slices = self.random_gamma(slices)
            slices = self.add_noise(slices)
            slices = self.signal_void(slices)
            # append stack
            if (
                self.max_num_slices is not None
                and sum(st.shape[0] for st in stacks) + slices.shape[0]
                >= self.max_num_slices
            ):
                break
            stacks.append(slices)
            stacks_no_psf.append(slices_no_psf)
            transforms.append(transform_init)
            transforms_gt.append(transform_target)
            positions.append(
                torch.arange(
                    slices.shape[0], dtype=slices.dtype, device=device
                )
                - slices.shape[0] // 2
            )
            if len(stacks) >= num_stacks:
                break
                # add stack index
        stacks_ids = np.random.choice(20, len(stacks), replace=False)
        positions = torch.cat(
            [
                torch.stack(
                    (positions[i], torch.full_like(positions[i], s_i)), -1
                )
                for i, s_i in enumerate(stacks_ids)
            ],
            0,
        )
        stacks = torch.cat(stacks, 0)
        stacks_no_psf = torch.cat(stacks_no_psf, 0)
        transforms = RigidTransform.cat(transforms)
        transforms_gt = RigidTransform.cat(transforms_gt)

        data["slice_shape"] = (ss, ss)
        data["volume_shape"] = volume_gt.shape[-3:]
        data["stacks"] = stacks
        data["stacks_no_psf"] = stacks_no_psf
        data["positions"] = positions
        data["transforms"] = transforms.matrix()
        data["transforms_angle"] = transforms
        data["transforms_gt"] = transforms_gt.matrix()
        data["transforms_gt_angle"] = transforms_gt

        data.pop("volume")
        return data

from fetalsyngen.generator.artifacts.utils import MergeParams, PerlinMergeParams, GaussianMergeParams, generate_fractal_noise_3d
class PSFReconstructor2:
    """
    Class that reconstructs the volume from the acquired slices using the PSF and the transforms given.

    Randomly applies: misregistration of a part of the slices, removal of a portion of the slices, merging of the volume with the ground truth according to a 3D MoG and smoothing of the volume.


    """

    def __init__(
        self,
        prob_misreg_slice: float,
        slices_misreg_ratio: float,
        prob_misreg_stack: float,
        txy: float,
        prob_merge: float,
        merge_params: MergeParams, 

        prob_smooth: float,
        prob_rm_slices: float,
        rm_slices_min: float,
        rm_slices_max: float,
    ):
        """
        Initialize the reconstructor with the given parameters.

        Args:
            prob_misreg_slice: Probability of misregistering a slice.
            slices_misreg_ratio: Ratio of slices to misregister.
            prob_misreg_stack: Probability of misregistering a stack.
            txy: Translation factor.
            prob_merge: Probability of merging the volume with the ground truth.
            merge_ngaussians_min: Minimum number of Gaussians for the merging.
            merge_ngaussians_max: Maximum number of Gaussians for the merging.
            prob_smooth: Probability of smoothing the volume.
            prob_rm_slices: Probability of removing slices.
            rm_slices_min: Minimum ratio of slices to remove.
            rm_slices_max: Maximum ratio of slices to remove.

        """
        self.prob_misreg_slice = prob_misreg_slice
        self.slices_misreg_ratio = slices_misreg_ratio
        self.prob_misreg_stack = prob_misreg_stack
        self.txy_stack = txy
        self.prob_merge = prob_merge
        self.merge_params = merge_params
        assert merge_params.merge_type in ["gaussian", "perlin"], (
                f"Merge type {merge_params.merge_type} not supported, "
                "only gaussian and perlin are supported."
                )
        self.prob_smooth = prob_smooth
        self.prob_rm_slices = prob_rm_slices
        self.rm_slices_min = rm_slices_min
        self.rm_slices_max = rm_slices_max

    def sample_seeds(self, genparams: dict = {}):
        """
        Sample the seeds for the randomization.

        Args:
            genparams: Dictionary containing the generation parameters.
        """
        self._smooth_volume_on = np.random.rand() < self.prob_smooth
        self._rm_slices_on = np.random.rand() < self.prob_rm_slices
        self._misreg_slice_on = np.random.rand() < self.prob_misreg_slice

        if "rm_slices_ratio" in genparams:
            self._rm_slices_ratio = genparams["rm_slices_ratio"]
        else:
            self._rm_slices_ratio = (
                np.random.uniform(self.rm_slices_min, self.rm_slices_max)
                if self._rm_slices_on
                else None
            )
        self._misreg_stack_on = []
        self._merge_volume_on = np.random.rand() < self.prob_merge
        if isinstance(self.merge_params, GaussianMergeParams):
            if "ngaussians_merge" in genparams:
                self._ngaussians_merge = genparams["ngaussians_merge"]
            else:
                self._ngaussians_merge = np.random.randint(
                    self.merge_ngaussians_min, self.merge_ngaussians_max
                )
        elif isinstance(self.merge_params, PerlinMergeParams):
            if "res" in genparams:
                self._res = genparams["res"]
            else:
                self._res = np.random.choice(self.merge_params.res_list)
            if "octave" in genparams:
                self._octave = genparams["octave"]
            else:
                self._octave = np.random.choice(self.merge_params.octave_list)

    def get_seeds(self):
        """
        Get the dictionary of the seeds used for randomization.
        """
        return {
            "smooth_volume_on": self._smooth_volume_on,
            "rm_slices_on": self._rm_slices_on,
            "rm_slices_ratio": self._rm_slices_ratio,
            "misreg_stack_on": self._misreg_stack_on,
            "misreg_slice_on": self._misreg_slice_on,
            "merge_volume_on": self._merge_volume_on,
            "ngaussians_merge": self._ngaussians_merge,
        }

    def smooth_volume(self, volume):
        """
        Smooth the volume using a 3x3x3 convolution kernel
        """
        if self._smooth_volume_on:
            return F.conv3d(
                volume,
                torch.ones(1, 1, 3, 3, 3, device=self.device) / 27,
                padding=1,
            )
        else:
            return volume

    def misregistration_trf(self, positions, base_axisangle):
        """
        Simulate misalignment with a registration error.

        Args:
            positions: The positions of the slices.
            base_axisangle: The base axis angle.

        Returns:
            RigidTransform: The misregistered transformation.
        """
        nslices = len(positions)
        rand_angle = torch.zeros((nslices, 6)).to(self.device)
        for pos in torch.unique(positions[:, 1]):
            self._misreg_stack_on.append(
                np.random.rand() < self.prob_misreg_stack
            )
            if not self._misreg_stack_on[-1]:
                continue
            idx = torch.where(positions[:, 1] == pos)[0]

            tx = torch.ones(len(idx)).to(self.device) * np.random.uniform(
                -self.txy_stack, self.txy_stack
            )
            ty = torch.ones(len(idx)).to(self.device) * np.random.uniform(
                -self.txy_stack, self.txy_stack
            )
            rand_angle[idx, 3:] = random_angle(
                len(idx), restricted=True, device=self.device
            )
            rand_angle[idx, :3] = torch.stack(
                (tx, ty, torch.zeros_like(tx)), -1
            )

        trf = RigidTransform(rand_angle, trans_first=True)

        return trf.compose(base_axisangle)

    def misregister_slices(self, trf, trf_gt):
        """
        Misregister a part of the slices based on the
        misregistration transform defined in `misregistration_trf`.

        Args:
            trf: The misregistered transformation.
            trf_gt: The ground truth transformation.
        """
        trf1 = trf.axisangle()
        trf2 = trf_gt.axisangle()
        if self._misreg_slice_on:
            idx_misreg = torch.randperm(trf2.shape[0])[
                : int(self.slices_misreg_ratio * trf2.shape[0])
            ]
            idx_misreg = idx_misreg[:1]
            trf2[idx_misreg] = trf1[idx_misreg]

        return RigidTransform(trf2, trans_first=True)


    def get_merging_weights(self, shape, vol_mask=None):
        """
        Get the merging weights for the volume.

        Args:
            shape: The shape of the volume.
            vol_mask: The volume mask.

        Returns:
            torch.Tensor: The merging weights.
        """

        if vol_mask is not None and self.merge_type=="gaussian":
            pos = torch.where(vol_mask.squeeze() > 0)
            idx = torch.randperm(pos[0].shape[0])[: self._ngaussians_merge]

            centers = [(pos[0][i], pos[1][i], pos[2][i]) for i in idx]
            # Tested for an image of size 256^3
            sigmas = [
                torch.clamp(20 + 10 * torch.randn(1), 5, 40).to(self.device)
                for _ in range(len(centers))
            ]
            weight = mog_3d_tensor(
                shape,
                centers=centers,
                sigmas=sigmas,
                device=self.device,
            ).view(1, 1, *shape)
            return weight
        elif self.merge_type == "perlin":
            weight = generate_fractal_noise_3d(
                shape,
                res=(self._res, self._res, self._res),
                octaves=self._octave,
                persistence=self.merge_params.persistence,
                lacunarity=self.merge_params.lacunarity,
                device=self.device,
            )
            return weight
        else:
            raise RuntimeError

    def merge_volumes(self, vol_mask, volume, volume_gt):
        """
        Merge the reconstructed volume with the ground truth according to a 3D mixture of Gaussians.
        This allows to simulate spatially varying artifacts.

        Args:
            vol_mask: The volume mask.
            volume: The reconstructed volume.
            volume_gt: The ground truth volume.
        """
        if self._merge_volume_on:
            weight = self.get_merging_weights(volume.shape, vol_mask)
            merged = weight * volume + (1 - weight) * volume_gt
            return merged, weight
        else:
            merged = volume

            return merged, torch.zeros_like(merged)

    def kept_slices_idx(self, nslices):
        """
        Get the indices of the slices that will be kept when removing a portion of the slices
        to be used for reconstruction.

        Args:
            nslices: The number of slices.

        Returns:
            torch.Tensor: The indices of the kept slices.
        """
        if self._rm_slices_on:
            # number of slices that will be kept.
            n = int(nslices * self._rm_slices_ratio)
            idx = torch.randperm(nslices)[n:]
            return idx
        else:
            return torch.arange(nslices)

    def recon_psf(self, data):
        """
        Reconstruct the volume using the PSF.

        Args:
            data: Dictionary containing the data.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The reconstructed volume and the mixture of Gaussians.
        """
        params = {
            "psf": data["psf_rec"],
            "slice_shape": data["slice_shape"],
            "interp_psf": True,
            "res_s": data["resolution_slice"],
            "res_r": data["resolution_recon"],
            "s_thick": data["slice_thickness"],
            "volume_shape": data["volume_shape"],
        }
        rec = partial(
            PSFreconstruction, slices_mask=None, vol_mask=None, params=params
        )
        return self.__recon_volume(data, rec)

    def __recon_volume(self, data, rec):
        """
        Reconstruct the volume using the given reconstruction function.

        Args:
            data: Dictionary containing the data.
            rec: The reconstruction function.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The reconstructed volume and the mixture of Gaussians.
        """

        self.sample_seeds()
        self.device = data["stacks"].device
        trf = self.misregister_slices(
            data["transforms_angle"], data["transforms_gt_angle"]
        )
        trf = self.misregistration_trf(data["positions"], trf)
        kept_idx = self.kept_slices_idx(data["stacks"].shape[0])
        volume = rec(trf.matrix()[kept_idx], data["stacks"][kept_idx])

        volume = self.smooth_volume(volume)
        mask = data["seg_gt"] > 0
        volume, mog = self.merge_volumes(mask, volume, data["volume_gt"])
        return volume, mog


class PSFReconstructor:
    """
    Class that reconstructs the volume from the acquired slices using the PSF and the transforms given.

    Randomly applies: misregistration of a part of the slices, removal of a portion of the slices, merging of the volume with the ground truth according to a 3D MoG and smoothing of the volume.


    """

    def __init__(
        self,
        prob_misreg_slice: float,
        slices_misreg_ratio: float,
        prob_misreg_stack: float,
        txy: float,
        prob_merge: float,
        merge_ngaussians_min: int,
        merge_ngaussians_max: int,
        prob_smooth: float,
        prob_rm_slices: float,
        rm_slices_min: float,
        rm_slices_max: float,
    ):
        """
        Initialize the reconstructor with the given parameters.

        Args:
            prob_misreg_slice: Probability of misregistering a slice.
            slices_misreg_ratio: Ratio of slices to misregister.
            prob_misreg_stack: Probability of misregistering a stack.
            txy: Translation factor.
            prob_merge: Probability of merging the volume with the ground truth.
            merge_ngaussians_min: Minimum number of Gaussians for the merging.
            merge_ngaussians_max: Maximum number of Gaussians for the merging.
            prob_smooth: Probability of smoothing the volume.
            prob_rm_slices: Probability of removing slices.
            rm_slices_min: Minimum ratio of slices to remove.
            rm_slices_max: Maximum ratio of slices to remove.

        """
        self.prob_misreg_slice = prob_misreg_slice
        self.slices_misreg_ratio = slices_misreg_ratio
        self.prob_misreg_stack = prob_misreg_stack
        self.txy_stack = txy
        self.prob_merge = prob_merge
        self.merge_ngaussians_min = merge_ngaussians_min
        self.merge_ngaussians_max = merge_ngaussians_max
        self.prob_smooth = prob_smooth
        self.prob_rm_slices = prob_rm_slices
        self.rm_slices_min = rm_slices_min
        self.rm_slices_max = rm_slices_max

    def sample_seeds(self, genparams: dict = {}):
        """
        Sample the seeds for the randomization.

        Args:
            genparams: Dictionary containing the generation parameters.
        """
        self._smooth_volume_on = np.random.rand() < self.prob_smooth
        self._rm_slices_on = np.random.rand() < self.prob_rm_slices
        self._misreg_slice_on = np.random.rand() < self.prob_misreg_slice

        if "rm_slices_ratio" in genparams:
            self._rm_slices_ratio = genparams["rm_slices_ratio"]
        else:
            self._rm_slices_ratio = (
                np.random.uniform(self.rm_slices_min, self.rm_slices_max)
                if self._rm_slices_on
                else None
            )
        self._misreg_stack_on = []
        self._merge_volume_on = np.random.rand() < self.prob_merge
        if "ngaussians_merge" in genparams:
            self._ngaussians_merge = genparams["ngaussians_merge"]
        else:
            self._ngaussians_merge = np.random.randint(
                self.merge_ngaussians_min, self.merge_ngaussians_max
            )

    def get_seeds(self):
        """
        Get the dictionary of the seeds used for randomization.
        """
        return {
            "smooth_volume_on": self._smooth_volume_on,
            "rm_slices_on": self._rm_slices_on,
            "rm_slices_ratio": self._rm_slices_ratio,
            "misreg_stack_on": self._misreg_stack_on,
            "misreg_slice_on": self._misreg_slice_on,
            "merge_volume_on": self._merge_volume_on,
            "ngaussians_merge": self._ngaussians_merge,
        }

    def smooth_volume(self, volume):
        """
        Smooth the volume using a 3x3x3 convolution kernel
        """
        if self._smooth_volume_on:
            return F.conv3d(
                volume,
                torch.ones(1, 1, 3, 3, 3, device=self.device) / 27,
                padding=1,
            )
        else:
            return volume

    def misregistration_trf(self, positions, base_axisangle):
        """
        Simulate misalignment with a registration error.

        Args:
            positions: The positions of the slices.
            base_axisangle: The base axis angle.

        Returns:
            RigidTransform: The misregistered transformation.
        """
        nslices = len(positions)
        rand_angle = torch.zeros((nslices, 6)).to(self.device)
        for pos in torch.unique(positions[:, 1]):
            self._misreg_stack_on.append(
                np.random.rand() < self.prob_misreg_stack
            )
            if not self._misreg_stack_on[-1]:
                continue
            idx = torch.where(positions[:, 1] == pos)[0]

            tx = torch.ones(len(idx)).to(self.device) * np.random.uniform(
                -self.txy_stack, self.txy_stack
            )
            ty = torch.ones(len(idx)).to(self.device) * np.random.uniform(
                -self.txy_stack, self.txy_stack
            )
            rand_angle[idx, 3:] = random_angle(
                len(idx), restricted=True, device=self.device
            )
            rand_angle[idx, :3] = torch.stack(
                (tx, ty, torch.zeros_like(tx)), -1
            )

        trf = RigidTransform(rand_angle, trans_first=True)

        return trf.compose(base_axisangle)

    def misregister_slices(self, trf, trf_gt):
        """
        Misregister a part of the slices based on the
        misregistration transform defined in `misregistration_trf`.

        Args:
            trf: The misregistered transformation.
            trf_gt: The ground truth transformation.
        """
        trf1 = trf.axisangle()
        trf2 = trf_gt.axisangle()
        if self._misreg_slice_on:
            idx_misreg = torch.randperm(trf2.shape[0])[
                : int(self.slices_misreg_ratio * trf2.shape[0])
            ]
            idx_misreg = idx_misreg[:1]
            trf2[idx_misreg] = trf1[idx_misreg]

        return RigidTransform(trf2, trans_first=True)

    def merge_volumes(self, vol_mask, volume, volume_gt):
        """
        Merge the reconstructed volume with the ground truth according to a 3D mixture of Gaussians.
        This allows to simulate spatially varying artifacts.

        Args:
            vol_mask: The volume mask.
            volume: The reconstructed volume.
            volume_gt: The ground truth volume.
        """
        if self._merge_volume_on:
            device = volume.device
            pos = torch.where(vol_mask.squeeze() > 0)
            idx = torch.randperm(pos[0].shape[0])[: self._ngaussians_merge]

            centers = [(pos[0][i], pos[1][i], pos[2][i]) for i in idx]
            # Tested for an image of size 256^3
            sigmas = [
                torch.clamp(20 + 10 * torch.randn(1), 5, 40).to(device)
                for _ in range(len(centers))
            ]
            weight = mog_3d_tensor(
                volume[0, 0].shape,
                centers=centers,
                sigmas=sigmas,
                device=device,
            ).view(1, 1, *volume.shape[2:])
            merged = weight * volume + (1 - weight) * volume_gt
            return merged, weight
        else:
            merged = volume

            return merged, torch.zeros_like(merged)

    def kept_slices_idx(self, nslices):
        """
        Get the indices of the slices that will be kept when removing a portion of the slices
        to be used for reconstruction.

        Args:
            nslices: The number of slices.

        Returns:
            torch.Tensor: The indices of the kept slices.
        """
        if self._rm_slices_on:
            # number of slices that will be kept.
            n = int(nslices * self._rm_slices_ratio)
            idx = torch.randperm(nslices)[n:]
            return idx
        else:
            return torch.arange(nslices)

    def recon_psf(self, data):
        """
        Reconstruct the volume using the PSF.

        Args:
            data: Dictionary containing the data.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The reconstructed volume and the mixture of Gaussians.
        """
        params = {
            "psf": data["psf_rec"],
            "slice_shape": data["slice_shape"],
            "interp_psf": True,
            "res_s": data["resolution_slice"],
            "res_r": data["resolution_recon"],
            "s_thick": data["slice_thickness"],
            "volume_shape": data["volume_shape"],
        }
        rec = partial(
            PSFreconstruction, slices_mask=None, vol_mask=None, params=params
        )
        return self.__recon_volume(data, rec)

    def __recon_volume(self, data, rec):
        """
        Reconstruct the volume using the given reconstruction function.

        Args:
            data: Dictionary containing the data.
            rec: The reconstruction function.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The reconstructed volume and the mixture of Gaussians.
        """

        self.sample_seeds()
        self.device = data["stacks"].device
        trf = self.misregister_slices(
            data["transforms_angle"], data["transforms_gt_angle"]
        )
        trf = self.misregistration_trf(data["positions"], trf)
        kept_idx = self.kept_slices_idx(data["stacks"].shape[0])
        volume = rec(trf.matrix()[kept_idx], data["stacks"][kept_idx])

        volume = self.smooth_volume(volume)
        mask = data["seg_gt"] > 0
        volume, mog = self.merge_volumes(mask, volume, data["volume_gt"])
        return volume, mog
