import torch
from fetalsyngen.intensity.seed_sampler import ImageFromSeeds
from fetalsyngen.deformation.affine_nonrigid import SpatialDeformation
import time
import SimpleITK as sitk

# TODO: Make different versions of the feta.yaml that work
# with FetalSimpleDataset and FetalSynthDataset as well as
# versions with/withoout transforms, with/without seeds
# and make an example notebook where all is called and illustrated
# time the synthetic generation


class SynthGen:

    def __init__(
        self,
        image_seed_generator: ImageFromSeeds,
        spatial_deform: SpatialDeformation,
        image_augmenter,
    ):
        self.intensity_generator = image_seed_generator
        self.spatial_deform = spatial_deform
        # seed loading parameters

    def sample(self, image, segmentation, seeds: torch.Tensor | None):
        generation_params = {}

        # 1. Generate intensity output
        # if seeds are passed, used them to generate the intensity
        # image randomly, otherwise skip this step and use the
        # original image
        start_time = time.time()

        if seeds is not None:
            seeds = self.intensity_generator.load_seeds(seeds)
            output = self.intensity_generator.sample_intensities(seeds)
            print("Intensity time: ", time.time() - start_time)
            # 0.21s per generation already and ~400mb of GPU
        else:
            if image is None:
                raise ValueError(
                    "If no seeds are passed, an image must be loaded to be used as intensity prior!"
                )
            output = image

        # 2. Spatially deform the inputs
        image, segmentation, output = self.spatial_deform.deform(
            image, segmentation, output
        )

        # print(f"Shape: {image.shape}, {segmentation.shape}, {output.shape}")
        # print(f"Dtype: {image.dtype}, {segmentation.dtype}, {output.dtype}")
        # print(f"Device: {image.device}, {segmentation.device}, {output.device}")
        # save image and the segmentation
        # sitk.WriteImage(sitk.GetImageFromArray(image), "image.nii")
        # sitk.WriteImage(
        #     sitk.GetImageFromArray(segmentation.cpu().numpy()), "segmentation.nii"
        # )
        # sitk.WriteImage(sitk.GetImageFromArray(output.cpu().numpy()), "output.nii")
        # print("Intensity+Deformation time: ", time.time() - start_time)

        # 3. Augment the output
        # don't change the image
        output = self.augment(output)
        # Resampling
        # Bias
        # Noise
        # Blur
        # Contrast-gama

        return output, segmentation, image, generation_params
