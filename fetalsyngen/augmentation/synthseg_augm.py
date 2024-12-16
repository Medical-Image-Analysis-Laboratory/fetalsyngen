import torch


class RandTransform:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def rand_uniform(min_val: float, max_val: float):
        return torch.rand(1) * (max_val - min_val) + min_val


class RandResample(RandTransform):

    def __init__(self, min_resampling_iso_res: float, max_resampling_iso_res: float):
        self.min_resampling_iso_res = min_resampling_iso_res
        self.max_resampling_iso_res = max_resampling_iso_res

    def __call__(self, *args, **kwargs):

class RandBiasField(RandTransform):
    pass


class RandNoise(RandTransform):
    pass


class RandBlur(RandTransform):
    pass


class RandGamma(RandTransform):
    pass
