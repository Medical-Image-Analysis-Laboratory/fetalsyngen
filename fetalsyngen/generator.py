class FetalSynthGen:
    def __init__(self):
        # intensity_generator: SeedGenerator | RealImageGenerator
        # deformation: DeformationClass
        pass


def sample(self, gen_input, gt_segmentation=None, gt_image=None)
	# gen_input is either seeds or image
	gen_output = intensity_generator.sample(gen_input)
	
    # deform
	gen_output, gt_segmentation, gt_image = deformation(gen_output, gt_segmentation, gt_image)
	
	# augment
	gen_output = augment(gen_output, generation_args)
	return gen_output, gt_segmentation, gt_image