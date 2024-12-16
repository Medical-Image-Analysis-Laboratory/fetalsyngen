import torch
import numpy as np
from fetalsyngen.brainid_utils import (
    fast_3D_interp_torch,
    myzoom_torch,
    make_affine_matrix,
    gaussian_blur_3d,
)


class SpatialDeformation:
    def __init__(
        self,
        max_rotation,
        max_shear,
        max_scaling,
        size,
        nonlinear_transform,
        nonlin_scale_min,
        nonlin_scale_max,
        nonlin_std_max,
        device="cuda",
    ):
        self.size = size  # 256, 256, 256

        # randaffine parameters
        self.max_rotation = max_rotation
        self.max_shear = max_shear
        self.max_scaling = max_scaling

        # nonlinear transform parameters
        self.nonlinear_transform = nonlinear_transform
        self.nonlin_scale_min = nonlin_scale_min
        self.nonlin_scale_max = nonlin_scale_max
        self.nonlin_std_max = nonlin_std_max

        self.device = device

        self._prepare_grid()

    def _prepare_grid(self):

        xx, yy, zz = np.meshgrid(
            range(self.size[0]),
            range(self.size[1]),
            range(self.size[2]),
            sparse=False,
            indexing="ij",
        )
        self.xx = torch.tensor(xx, dtype=torch.float, device=self.device)
        self.yy = torch.tensor(yy, dtype=torch.float, device=self.device)
        self.zz = torch.tensor(zz, dtype=torch.float, device=self.device)
        self.c = torch.tensor(
            (np.array(self.size) - 1) / 2,
            dtype=torch.float,
            device=self.device,
        )
        self.xc = self.xx - self.c[0]
        self.yc = self.yy - self.c[1]
        self.zc = self.zz - self.c[2]

    def deform(self, image, segmentation, output):
        image_shape = output.shape
        flip = np.random.rand() < 0.5  # TODO: Change this to a parameter

        xx2, yy2, zz2, x1, y1, z1, x2, y2, z2 = self.generate_deformation(
            image_shape, random_shift=True
        )
        # flip the image if nessesary
        if flip:
            segmentation = torch.flip(segmentation, [0])
            output = torch.flip(output, [0])
            image = torch.flip(image, [0]) if image is not None else None

        output = fast_3D_interp_torch(output, xx2, yy2, zz2, "linear")
        segmentation = fast_3D_interp_torch(
            segmentation.to(self.device), xx2, yy2, zz2, "nearest"
        )
        if image is not None:
            image = fast_3D_interp_torch(image.to(self.device), xx2, yy2, zz2, "linear")
        return image, segmentation, output

    def generate_deformation(self, image_shape, random_shift=True):

        # sample affine deformation
        A, c2 = self.random_affine_transform(
            image_shape,
            self.max_rotation,
            self.max_shear,
            self.max_scaling,
            random_shift,
        )

        # sample nonlinear deformation
        if self.nonlinear_transform:
            F = self.random_nonlinear_transform(
                self.nonlin_scale_min,
                self.nonlin_scale_max,
                self.nonlin_std_max,
            )
        else:
            F = None

        # deform the images
        xx2, yy2, zz2, x1, y1, z1, x2, y2, z2 = self.deform_image(image_shape, A, c2, F)

        return xx2, yy2, zz2, x1, y1, z1, x2, y2, z2

    def random_affine_transform(
        self, shp, max_rotation, max_shear, max_scaling, random_shift=True
    ):
        rotations = (
            (2 * max_rotation * np.random.rand(3) - max_rotation) / 180.0 * np.pi
        )

        shears = 2 * max_shear * np.random.rand(3) - max_shear
        scalings = 1 + (2 * max_scaling * np.random.rand(3) - max_scaling)
        # we divide distance maps by this, not perfect, but better than nothing
        A = torch.tensor(
            make_affine_matrix(rotations, shears, scalings),
            dtype=torch.float,
            device=self.device,
        )
        # sample center
        if random_shift:
            max_shift = (
                torch.tensor(
                    np.array(shp[0:3]) - self.size,
                    dtype=torch.float,
                    device=self.device,
                )
            ) / 2
            max_shift[max_shift < 0] = 0
            c2 = torch.tensor(
                (np.array(shp[0:3]) - 1) / 2,
                dtype=torch.float,
                device=self.device,
            ) + (
                2 * (max_shift * torch.rand(3, dtype=float, device=self.device))
                - max_shift
            )
        else:
            c2 = torch.tensor(
                (np.array(shp[0:3]) - 1) / 2,
                dtype=torch.float,
                device=self.device,
            )
        return A, c2

    def random_nonlinear_transform(
        self,
        nonlin_scale_min,
        nonlin_scale_max,
        nonlin_std_max,
    ):

        nonlin_scale = nonlin_scale_min + np.random.rand(1) * (
            nonlin_scale_max - nonlin_scale_min
        )
        size_F_small = np.round(nonlin_scale * np.array(self.size)).astype(int).tolist()
        nonlin_std = nonlin_std_max * np.random.rand()
        Fsmall = nonlin_std * torch.randn(
            [*size_F_small, 3], dtype=torch.float, device=self.device
        )
        F = myzoom_torch(Fsmall, np.array(self.size) / size_F_small)

        return F

    def deform_image(self, shp, A, c2, F):
        if F is not None:
            # deform the images (we do nonlinear "first" ie after so we can do heavy coronal deformations in photo mode)
            xx1 = self.xc + F[:, :, :, 0]
            yy1 = self.yc + F[:, :, :, 1]
            zz1 = self.zc + F[:, :, :, 2]
        else:
            xx1 = self.xc
            yy1 = self.yc
            zz1 = self.zc

        xx2 = A[0, 0] * xx1 + A[0, 1] * yy1 + A[0, 2] * zz1 + c2[0]
        yy2 = A[1, 0] * xx1 + A[1, 1] * yy1 + A[1, 2] * zz1 + c2[1]
        zz2 = A[2, 0] * xx1 + A[2, 1] * yy1 + A[2, 2] * zz1 + c2[2]
        xx2[xx2 < 0] = 0
        yy2[yy2 < 0] = 0
        zz2[zz2 < 0] = 0
        xx2[xx2 > (shp[0] - 1)] = shp[0] - 1
        yy2[yy2 > (shp[1] - 1)] = shp[1] - 1
        zz2[zz2 > (shp[2] - 1)] = shp[2] - 1

        # Get the margins for reading images
        x1 = torch.floor(torch.min(xx2))
        y1 = torch.floor(torch.min(yy2))
        z1 = torch.floor(torch.min(zz2))
        x2 = 1 + torch.ceil(torch.max(xx2))
        y2 = 1 + torch.ceil(torch.max(yy2))
        z2 = 1 + torch.ceil(torch.max(zz2))
        xx2 -= x1
        yy2 -= y1
        zz2 -= z1

        x1 = x1.cpu().numpy().astype(int)
        y1 = y1.cpu().numpy().astype(int)
        z1 = z1.cpu().numpy().astype(int)
        x2 = x2.cpu().numpy().astype(int)
        y2 = y2.cpu().numpy().astype(int)
        z2 = z2.cpu().numpy().astype(int)
        return xx2, yy2, zz2, x1, y1, z1, x2, y2, z2
