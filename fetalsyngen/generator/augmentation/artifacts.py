from .synthseg import RandTransform
import numpy as np
import torch
from fetalsyngen.generator.artifacts.simulate_reco import (
    Scanner,
    PSFReconstructor,
)
import copy
from fetalsyngen.generator.artifacts.utils import (
    gaussian_blur_3d,
    mog_3d_tensor,
    apply_kernel,
    erode,
    dilate,
    ScannerParams,
    ReconParams,
)
from skimage.morphology import ball
from dataclasses import asdict


class BlurCortex(RandTransform):

    def __init__(
        self,
        prob: float,
        cortex_label: int,
        nblur_min: int,
        nblur_max: int,
        sigma_gamma_loc: int = 3,
        sigma_gamma_scale: int = 1,
        std_blur_shape: int = 2,
        std_blur_scale: int = 1,
    ):
        self.prob = prob
        self.cortex_label = cortex_label
        self.nblur_min = nblur_min
        self.nblur_max = nblur_max
        self.sigma_gamma_loc = sigma_gamma_loc
        self.sigma_gamma_scale = sigma_gamma_scale
        self.std_blur_shape = std_blur_shape
        self.std_blur_scale = std_blur_scale

    def blur_proba(self, shape, cortex, device):
        x, y, z = shape
        # Blurring is more likely to happen in the frontal lobe
        cortex_prob = mog_3d_tensor(
            shape,
            [(0, y, z // 2), (x, y, z // 2)],
            [x // 5, y // 5],
            device,
        )
        idx_cortex = torch.where(cortex > 0)
        cortex_prob = cortex_prob[idx_cortex]
        cortex_prob = cortex_prob / cortex_prob.sum()
        return cortex_prob

    def __call__(self, output, seg, device, genparams: dict = {}, **kwargs):

        if np.random.rand() < self.prob or len(genparams.keys()) > 0:
            nblur = (
                np.random.randint(self.nblur_min, self.nblur_max)
                if "nblur" not in genparams.keys()
                else genparams["nblur"]
            )
            std_blurs = np.random.gamma(
                self.std_blur_shape, self.std_blur_scale, 3
            )

            cortex = seg == self.cortex_label
            cortex_prob = self.blur_proba(output.shape, cortex, device)
            idx = torch.multinomial(cortex_prob, nblur)

            idx_cortex = torch.where(cortex > 0)
            centers = [(idx_cortex[i][id] for i in range(3)) for id in idx]

            # Spatial merging parameters.
            sigmas = np.random.gamma(
                self.sigma_gamma_loc, self.sigma_gamma_scale, (nblur, 3)
            )
            gaussian = mog_3d_tensor(
                output.shape,
                centers=centers,
                sigmas=sigmas,
                device=output.device,
            )

            # Generate the blurred image

            # USELESS TO SPECIFY DEVICE. SHOULD BE
            # THE SAME AS OUTPUT
            output_blur = gaussian_blur_3d(
                output.float(), stds=std_blurs, device=output.device
            )
            output = output * (1 - gaussian) + output_blur * gaussian
            return output, {
                "nblur": nblur,
            }
        else:
            return output, {
                "nblur": None,
            }


class StructNoise(RandTransform):
    ### TO REFACTOR: THIS IS PERLIN NOISE
    def __init__(
        self,
        prob: float,
        wm_label: int,
        std_min: float,
        std_max: float,
        nloc_min: int,
        nloc_max: int,
        nstages_min: int = 1,
        nstages_max: int = 5,
        sigma_mu: int = 25,
        sigma_std: int = 5,
    ):
        self.prob = prob
        self.wm_label = wm_label
        self.nstages_min = nstages_min
        self.nstages_max = nstages_max
        self.std_min = std_min
        self.std_max = std_max
        self.nloc_min = nloc_min
        self.nloc_max = nloc_max
        self.sigma_mu = sigma_mu
        self.sigma_std = sigma_std

    def __call__(self, output, seg, device, genparams: dict = {}, **kwargs):

        if np.random.rand() < self.prob or "nloc" in genparams.keys():
            ## Parameters
            nstages = (
                np.random.randint(self.nstages_min, self.nstages_max)
                if "nstages" not in genparams
                else genparams["nstages"]
            )
            noise_std = (
                self.std_min + (self.std_max - self.std_min) * np.random.rand()
            )
            nloc = (
                np.random.randint(
                    self.nloc_min,
                    self.nloc_max,
                )
                if "nloc" not in genparams
                else genparams["nloc"]
            )
            ##

            wm = seg == self.wm_label
            idx_wm = torch.nonzero(wm, as_tuple=True)
            idx = torch.randint(0, len(idx_wm[0]), (nloc,))

            # Add multiscale noise. Start with a small tensor and add the noise to it.
            lr_gaussian_noise = torch.zeros(
                [i // 2**nstages for i in output.shape]
            ).to(device)

            for k in range(nstages):
                shape = [i // 2 ** (nstages - k) for i in output.shape]
                next_shape = [
                    i // 2 ** (nstages - 1 - k) for i in output.shape
                ]
                lr_gaussian_noise += torch.randn(shape).to(device)
                lr_gaussian_noise = torch.nn.functional.interpolate(
                    lr_gaussian_noise.unsqueeze(0).unsqueeze(0),
                    size=next_shape,
                    mode="trilinear",
                    align_corners=False,
                ).squeeze()

            lr_gaussian_noise = lr_gaussian_noise / torch.max(
                abs(lr_gaussian_noise)
            )
            output_noisy = torch.clamp(
                output + noise_std * lr_gaussian_noise, 0, output.max() * 2
            )

            sigmas = (
                (
                    torch.clamp(
                        self.sigma_mu + self.sigma_std * torch.randn(len(idx)),
                        1,
                        40,
                    )
                )
                .cpu()
                .numpy()
            )
            centers = [
                (
                    idx_wm[0][id].item(),
                    idx_wm[1][id].item(),
                    idx_wm[2][id].item(),
                )
                for id in idx
            ]
            import pdb

            pdb.set_trace()
            gaussian = mog_3d_tensor(
                output.shape, centers=centers, sigmas=sigmas, device=device
            )

            output = gaussian * output_noisy + (1 - gaussian) * output

            args = {
                "nstages": nstages,
                "noise_std": noise_std,
                "nloc": nloc,
            }
            return output, args
        else:
            return output, {
                "nstages": None,
                "noise_std": None,
                "nloc": None,
            }


class SimulateMotion(RandTransform):

    def __init__(
        self,
        prob: float,
        scanner_params: ScannerParams,
        recon_params: ReconParams,
    ):

        self.scanner_args = scanner_params
        self.recon_args = recon_params
        self.prob = prob

    def __call__(self, output, seg, device, genparams: dict = {}, **kwargs):
        # def _artifact_simulate_motion(self, im, seg, generator_params, res):
        if np.random.rand() < self.prob:
            device = output.device
            dshape = (1, 1, *output.shape[-3:])
            res = kwargs["resolution"]
            res_ = np.float64(res[0])

            d = {
                "resolution": res_,
                "volume": output.view(dshape).float().to(device),
                "mask": (seg > 0).view(dshape).float().to(device),
                "seg": seg.view(dshape).float().to(device),
                "affine": torch.diag(torch.tensor(list(res) + [1])).to(device),
                "threshold": 0.1,
            }
            self.scanner_args.resolution_recon = res_
            scanner = Scanner(**asdict(self.scanner_args))
            d_scan = scanner.scan(d)
            # Copy self recon_args

            # recon_args = copy.deepcopy(self.recon_args)
            # recon_args["prob_merge"] = generator_params.rec_prob_merge
            # recon_args["merge_ngaussians_min"] = (
            #     generator_params.rec_merge_ngaussians_min
            # )
            # recon_args["merge_ngaussians_max"] = (
            #     generator_params.rec_merge_ngaussians_max
            # )
            recon = PSFReconstructor(**asdict(self.recon_args))
            output, _ = recon.recon_psf(d_scan)

            self.metadata.update(
                {
                    "resolution_recon": d_scan["resolution_recon"],
                    "resolution_slice": d_scan["resolution_slice"],
                    "slice_thickness": d_scan["slice_thickness"],
                    "gap": d_scan["gap"],
                    "nstacks": len(torch.unique(d_scan["positions"][:, 1])),
                }
            )
            self.metadata.update(recon.get_seeds())

            return output.squeeze(), {}
        else:
            return output, {}


class SimulatedBoundaries(RandTransform):
    def __init__(
        self, prob_no_mask: float, prob_halo: float, prob_fuzzy: float
    ):
        self.prob_no_mask = prob_no_mask
        self.prob_halo = prob_halo
        self.prob_fuzzy = prob_fuzzy
        self.reset_seeds()

    def reset_seeds(self):
        self.no_mask_on = None
        self.halo_on = None
        self.halo_radius = None
        self.fuzzy_on = None
        self.n_generate_fuzzy = None
        self.n_centers = None
        self.base_sigma = None

    def sample_seeds(self):
        self.reset_seeds()
        self.no_mask_on = np.random.rand() < self.prob_no_mask
        if not self.no_mask_on:
            self.halo_on = np.random.rand() < self.prob_halo
            if self.halo_on:
                self.halo_radius = np.random.randint(5, 15)
            self.fuzzy_on = np.random.rand() < self.prob_fuzzy
            if self.fuzzy_on:
                self.n_generate_fuzzy = np.random.randint(2, 5)
                self.n_centers = np.random.poisson(100)
                self.base_sigma = np.random.poisson(8)

    def build_halo(self, mask, radius):

        device = mask.device
        kernel = (
            torch.tensor(ball(radius))
            .float()
            .to(device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        mask = mask.float().view(1, 1, *mask.shape[-3:])
        mask = torch.nn.functional.conv3d(mask, kernel, padding="same")
        return (mask > 0).int()

    def generate_fuzzy_boundaries(
        self, mask, kernel_size=7, threshold_filter=3
    ):
        shape = mask.shape
        diff = (dilate(mask, kernel_size) - mask).view(shape[-3:])
        non_zero = diff.nonzero(as_tuple=True)
        idx = torch.randperm(len(non_zero[0]))[: int(len(non_zero[0]) * 0.9)]
        idx = (non_zero[0][idx], non_zero[1][idx], non_zero[2][idx])
        diff[idx] = 0

        dsamp = (apply_kernel(diff).squeeze() > threshold_filter).bool()
        closing = erode(dilate(torch.clamp(mask + dsamp, 0, 1), 5), 5)
        return closing.view(shape)

    def __call__(self, output, seg, device, genparams: dict = {}, **kwargs):
        device = seg.device
        mask = (seg > 0).int()
        mask = mask.clone()

        self.sample_seeds()
        if self.no_mask_on:
            return output, mask
        if self.halo_on:
            mask = self.build_halo(mask, self.halo_radius)
        if self.fuzzy_on:
            # Generate fuzzy boundaries for the mask
            mask_modif = mask.clone()
            for _ in range(self.n_generate_fuzzy):
                mask_modif = self.generate_fuzzy_boundaries(mask_modif)

            # Sample centers in the voxels that have been added
            # with a MoG

            surf = torch.where((mask_modif - mask).squeeze() > 0)
            idx = torch.randperm(surf[0].shape[0])[: self.n_centers]
            centers = [(surf[0][i], surf[1][i], surf[2][i]) for i in idx]
            sigmas = [
                self.base_sigma + 10 * np.random.beta(2, 5)
                for _ in range(len(centers))
            ]
            mog = mog_3d_tensor(
                mask_modif.shape[-3:],
                centers=centers,
                sigmas=sigmas,
                device=device,
            ).view(1, 1, *mask_modif.shape[-3:])

            # Generate the probability map for the surface

            surf_proba = torch.zeros_like(mog[0, 0]).float()
            surf_proba[surf] = mog[0, 0][surf]
            # Generate kernel_size-1 x n_generate_fuzzy -1 dilations
            # Roughly matches the width of the generated halo
            n_dilate = 6 * (self.n_generate_fuzzy - 1)

            # Then, generate more realistic boundaries by making the
            # boundary of the bask more or less large according to the
            # probability map.
            dilate_stack = [mask.view(1, 1, *mask.shape[-3:])] * 2
            for i in range(n_dilate - 2):
                dilate_stack.append(self.build_halo(dilate_stack[-1], 1))

            # Generate a stack of dilations intersected with the mask
            dilate_stack = torch.stack(dilate_stack, 0)[
                :, 0, 0
            ] * mask_modif.view(1, *mask_modif.shape[-3:])

            surf_proba = torch.clamp(
                (surf_proba * len(dilate_stack) - 1).round().int(), 0, None
            )

            # Generate the final mask with the fuzzily generated boundaries
            # and also randomized halos.
            one_hot = torch.nn.functional.one_hot(
                surf_proba.to(torch.int64), num_classes=len(dilate_stack)
            ).int()
            dilate_stack = dilate_stack.permute(1, 2, 3, 0).int()
            mask = (one_hot * dilate_stack).sum(-1).unsqueeze(0).unsqueeze(0)
        return output * mask, {}
