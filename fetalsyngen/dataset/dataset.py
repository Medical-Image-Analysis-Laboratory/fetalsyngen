from collections import defaultdict
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import torch
from fetalsynthgen.generation import SynthGenerator, IDSynthGenerator
from pathlib import Path
from fetalsynthgen.definitions import GeneratorParams
from monai.transforms import (
    LoadImage,
    ScaleIntensityd,
    SignalFillEmptyd,
    CropForegroundd,
    SpatialPadd,
    Orientation,
    Spacingd,
    CenterSpatialCropd,
)
from monai.transforms import Compose
import os


class FetalDataset:
    """Abstract class defining a dataset for loading fetal data."""

    def __init__(
        self,
        bids_path: str,
        sub_list: list[str] | None,
    ):
        """
        Args:
            bids_path (str): Path to the bids folder with the data.
            sub_list (list[str]): List of the subjects to use.

        """
        super().__init__()

        self.bids_path = Path(bids_path)
        self.subjects = sub_list
        if self.subjects is None:
            self.subjects = [x.name for x in self.bids_path.glob("sub-*")]
        self.sub_ses = [
            (x, y) for x in self.subjects for y in self._get_ses(self.bids_path, x)
        ]

    def _sub_ses_string(self, sub, ses):
        return f"{sub}_{ses}" if ses is not None else sub

    def _sub_ses_idx(self, idx):
        sub, ses = self.sub_ses[idx]
        return self._sub_ses_string(sub, ses)

    def _get_ses(self, bids_path, sub):
        """Get the session names for the subject."""
        sub_path = bids_path / sub
        ses_dir = [x for x in sub_path.iterdir() if x.is_dir()]
        ses = []
        for s in ses_dir:
            if "anat" in s.name:
                ses.append(None)
            else:
                ses.append(s.name)

        return sorted(ses, key=lambda x: x or "")

    def _get_pattern(self, sub, ses, suffix, extension=".nii.gz"):
        """Get the pattern for the file name."""
        if ses is None:
            return f"{sub}/anat/{sub}*_{suffix}{extension}"
        else:
            return f"{sub}/{ses}/anat/{sub}_{ses}*_{suffix}{extension}"

    def _load_bids_path(self, path, suffix, load_paths=True):
        """
        "Check that for a given path, all subjects have a file with the provided suffix
        """
        files_paths = []
        for sub, ses in self.sub_ses:
            pattern = self._get_pattern(sub, ses, suffix)
            files = list(path.glob(pattern))
            if len(files) == 0:
                raise FileNotFoundError(
                    f"No files found for requested subject {sub} in {path} "
                    f"({pattern} returned nothing)"
                )
            elif len(files) > 1:
                raise RuntimeError(
                    f"Multiple files found for requested subject {sub} in {path} "
                    f"({pattern} returned {files})"
                )
            files_paths.append(files[0])

        if load_paths:
            return [nib.load(f) for f in files_paths]
        else:
            return files_paths

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        raise NotImplementedError(
            "This method should be implemented in the child class."
        )


# # in all three classes in get_data() have a scaling step just after loading the image
# class FetalSynthDataset(FetalDataset):
# 	def __inint__():
# 		synth_gen = FetalSynthGen()
# 	# file loading
# 	def __getitem__():
# 		# load the seeds
# 		image, seeds, segmentation = get_data()
# 		gen_output, segmentation, image= synth_gen.sample(seeds, segmentation, image)
# 		return gen_output, segmentation, image

# class FetalAugmDataset(FetalDataset):
# 	def __init__():
# 			synth_gen = FetalSynthGen()

# 	def __getitem__():
# 		image, segmentation = get_data()
# 		gen_output, segmentation, image = synth_gen.sample(image, segmentation, image)
# 		return gen_output, segmentation, image

# class FetalSimpleDataset(FetalDataset)

# 		image, segmentation = get_data()
# 		return image, segmentation, image
