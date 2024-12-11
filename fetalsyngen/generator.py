import torch


class SynthGen:
    def __init__(self, param1):

        self.intensity_generator = param1

    def sample(self, image, segmentation, seeds: torch.Tensor | None):
        return 1, 2, 3, 4
        # TODO: Clarify with Thomas should the image be augmented or not when output
        # is image

        # 1. Generate intensity output
        # if seeds are passed, used them to generate the intensity
        # image randomly, otherwise skip this step and use the
        # original image
        # if seeds is not None:
        #     output = self.intensity_generator(seeds)
        # else:
        #     output = image

        # # 2. Spatially deform the inputs
        # image, segmentation, output = self.spatial_deform(image, segmentation, output)

        # # 3. Augment the inputs
        # image, segmentation, output = self.augment(image, segmentation, output)

        # return output, segmentation, image
