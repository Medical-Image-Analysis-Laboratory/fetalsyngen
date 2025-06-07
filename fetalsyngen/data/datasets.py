from collections import defaultdict
from pathlib import Path
from monai.transforms import (
    ScaleIntensity,
    Orientation,
)
from monai.transforms import Compose
from fetalsyngen.generator.model import FetalSynthGen
from hydra.utils import instantiate
from fetalsyngen.utils.image_reading import SimpleITKReader
import time
import torch
import numpy as np
from monai.data import MetaTensor
import nibabel as nib
import numpy as np
import os


class FetalDataset:
    """Abstract class defining a dataset for loading fetal data."""

    def __init__(
        self,
        bids_path: str,
        sub_list: list[str] | None,
        bids_path_img_ending: str,
        bids_path_segm_ending: str,
    ) -> dict:
        """
        Args:
            bids_path: Path to the bids folder with the data.
            sub_list: List of the subjects to use. If None, all subjects are used.


        """
        super().__init__()

        self.bids_path = Path(bids_path)
        self.subjects = self.find_subjects(sub_list)
        if self.subjects is None:
            self.subjects = [x.name for x in self.bids_path.glob("sub-*")]
        self.sub_ses = [
            (x, y) for x in self.subjects for y in self._get_ses(self.bids_path, x)
        ]
        self.loader = SimpleITKReader()
        self.scaler = ScaleIntensity(minv=0, maxv=1)

        self.orientation = Orientation(axcodes="RAS")

        self.img_paths, self.segm_paths = self._load_bids_path(self.bids_path)
        #self.segm_paths = self._load_bids_path(self.bids_path, self.bids_path_segm_ending)

    def find_subjects(self, sub_list):
        subj_found = [x.name for x in Path(self.bids_path).glob("sub-*")]
        return list(set(subj_found) & set(sub_list)) if sub_list is not None else None

    def _sub_ses_string(self, sub, ses):
        return f"{sub}_{ses}" if ses is not None else sub

    def _sub_ses_idx(self, idx):
        sub, ses = self.sub_ses[idx]
        return self._sub_ses_string(sub, ses)

    def _get_ses(self, bids_path, sub):
        """Get the session names for the subject."""
        sub_path = bids_path / sub
        if not sub_path.exists():
            print(f"Subject {sub} not found at {sub_path}")

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
            return f"{sub}/{ses}/anat/{sub}*_{suffix}{extension}"
        
    def _load_metalabel_path(self, path, suffix):
        """
        "Check that for a given path, all subjects have a file with the provided suffix
        """
        files_paths = []
        for sub, ses in self.sub_ses:
            pattern = self._get_pattern(sub, ses, suffix)
            files = list(path.glob(pattern))
            if len(files) == 0:
                print(
                    f"No files found for requested subject {sub} in {path} "
                    f"({pattern} returned nothing)"
                )
            elif len(files) > 1:
                print(
                    f"Multiple files found for requested subject {sub} in {path} "
                    f"({pattern} returned {files})"
                )
            else: files_paths.append(files[0])

        return files_paths

    def _load_bids_path(self, path):
        """
        Check that for a given path, all subjects have a file with the provided suffix
        (either image or segmentation). If no segmentation is found, the image will be skipped.
        """
        files_paths = []
        skip_subjects = set()  # A set to keep track of subjects to skip

        for sub, ses in self.sub_ses:
            pattern = self._get_pattern(sub, ses, self.bids_path_segm_ending)
            files = list(path.glob(pattern))
            if len(files) == 0:
                # If no segmentation files found, print and skip the subject for both image and segmentation
                print(
                    f"No files found for requested subject {sub} in {path} "
                    f"({pattern} returned nothing)"
                )
                skip_subjects.add((sub, ses))  # Mark this subject-session pair to be skipped
            elif len(files) > 1:
                # If multiple files found, print a warning
                cc_files = [f for f in files if "CC" in os.path.basename(f)]
                if cc_files: file = cc_files[0]  # Use the first matching "CC" file
                else: file = files[0] 
                files_paths.append(file)
                print(
                    f"Multiple files found for requested subject {sub} in {path} "
                    f"({file} was selected.)"
                )
            else:
                files_paths.append(files[0])

        # Now, we need to ensure that if we skip a subject due to missing segmentation,
        # we also skip the corresponding image paths
        # Filter out skipped subjects before processing
        self.sub_ses = [(sub, ses) for sub, ses in self.sub_ses if (sub, ses) not in skip_subjects]

        img_paths = []
        for sub, ses in self.sub_ses:
            if (sub, ses) not in skip_subjects:  # Only load image files if not skipped
                pattern = self._get_pattern(sub, ses, self.bids_path_img_ending)
                files = list(path.glob(pattern))
                if len(files) == 0:
                    print(
                        f"No image files found for requested subject {sub} in {path} "
                        f"({pattern} returned nothing)"
                    )
                elif len(files) > 1:
                    print(
                        f"Multiple image files found for requested subject {sub} in {path} "
                        f"({pattern} returned {files})"
                    )
                else:
                    img_paths.append(files[0])

        return img_paths, files_paths  # Return both image and segmentation file paths


    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        raise NotImplementedError(
            "This method should be implemented in the child class."
        )


class FetalTestDataset(FetalDataset):
    """Dataset class for loading fetal images offline.
    Used to load test/validation data.

    Use the `transforms` argument to pass additional processing steps
    (scaling, resampling, cropping, etc.).
    """

    def __init__(
        self,
        bids_path: str,
        sub_list: list[str] | None,
        bids_path_img_ending: str,
        bids_path_segm_ending: str,
        transforms: Compose | None = None,
    ):
        """
        Args:
            bids_path: Path to the bids folder with the data.
            sub_list: List of the subjects to use. If None, all subjects are used.
            transforms: Compose object with the transformations to apply.
                Default is None, no transformations are applied.

        !!! Note
            We highle recommend using the `transforms` arguments with at
            least the re-oriented transform to RAS and the intensity scaling
            to `[0, 1]` to ensure the data consistency.

            See [inference.yaml](https://github.com/Medical-Image-Analysis-Laboratory/fetalsyngen/blob/dev/configs/dataset/transforms/inference.yaml) for an example of the transforms configuration.
        """
        self.bids_path_img_ending = bids_path_img_ending
        self.bids_path_segm_ending = bids_path_segm_ending
        super().__init__(bids_path, sub_list, bids_path_img_ending, bids_path_segm_ending, )
        self.transforms = transforms

    def _load_data(self, idx):
        # load the image and segmentation
        image = self.loader(self.img_paths[idx])
        segm = self.loader(self.segm_paths[idx])
        if len(image.shape) == 3:
            # add channel dimension
            image = image.unsqueeze(0)
            segm = segm.unsqueeze(0)
        elif len(image.shape) != 4:
            raise ValueError(f"Expected 3D or 4D image, got {len(image.shape)}D image.")

        # transform name into a single string otherwise collate fails
        name = self.sub_ses[idx]
        name = self._sub_ses_string(name[0], ses=name[1])

        return {"image": image, "label": segm.long(), "name": name}

    def __getitem__(self, idx) -> dict:
        """
        Returns:
            Dictionary with the `image` , `label` and the `name`
                keys. `image` and `label` are  `torch.float32`
                [`monai.data.meta_tensor.MetaTensor`](https://docs.monai.io/en/stable/data.html#metatensor)
                instances  with dimensions `(1, H, W, D)` and `name` is a string
                of a format `sub_ses` where `sub` is the subject name
                and `ses` is the session name.


        """
        data = self._load_data(idx)

        if self.transforms:
            data = self.transforms(data)
        data["label"] = data["label"].long()
        return data

    def reverse_transform(self, data: dict) -> dict:
        """Reverse the transformations applied to the data.

        Args:
            data: Dictionary with the `image` and `label` keys,
                like the one returned by the `__getitem__` method.

        Returns:
            Dictionary with the `image` and `label` keys where
                the transformations are reversed.
        """
        if self.transforms:
            data = self.transforms.inverse(data)
        return data


class FetalSynthDataset(FetalDataset):
    """Dataset class for generating/augmenting on-the-fly fetal images" """

    def __init__(
        self,
        bids_path: str,
        generator: FetalSynthGen,
        seed_path: str | None,
        seed_csf_path: str | None,
        sub_list: list[str] | None,
        bids_path_img_ending: str,
        bids_path_segm_ending: str,
        load_image: bool = False,
        image_as_intensity: bool = False,
    ):
        """

        Args:
            bids_path: Path to the bids-formatted folder with the data.
            seed_path: Path to the folder with the seeds to use for
                intensity sampling. See `scripts/seed_generation.py`
                for details on the data formatting. If seed_path is None,
                the intensity  sampling step is skipped and the output image
                intensities will be based on the input image.
            generator: a class object defining a generator to use.
            sub_list: List of the subjects to use. If None, all subjects are used.
            load_image: If **True**, the image is loaded and passed to the generator,
                where it can be used as the intensity prior instead of a random
                intensity sampling or spatially deformed with the same transformation
                field as segmentation and the syntehtic image. Default is **False**.
            image_as_intensity: If **True**, the image is used as the intensity prior,
                instead of sampling the intensities from the seeds. Default is **False**.
        """
        self.bids_path_img_ending = bids_path_img_ending
        self.bids_path_segm_ending = bids_path_segm_ending
        super().__init__(bids_path, sub_list, bids_path_img_ending, bids_path_segm_ending)
        self.seed_path = Path(seed_path) if isinstance(seed_path, str) else None
        self.seed_csf_path = Path(seed_csf_path) if isinstance(seed_csf_path, str) else None
        self.load_image = load_image
        self.generator = generator
        self.image_as_intensity = image_as_intensity

        # parse seeds paths
        if not self.image_as_intensity and isinstance(self.seed_path, Path):
            if not self.seed_path.exists():
                raise FileNotFoundError(
                    f"Provided seed path {self.seed_path} does not exist."
                )
            else:
                self._load_seed_path()
        # parse seeds paths
        if not self.image_as_intensity and isinstance(self.seed_csf_path, Path):
            if not self.seed_csf_path.exists():
                raise FileNotFoundError(
                    f"Provided seed csf path {self.seed_csf_path} does not exist."
                )
            else:
                self._load_seed_csf_path()

    def _load_seed_path(self):
        """Load the seeds for the subjects."""
        self.seed_paths = {
            self._sub_ses_string(sub, ses): defaultdict(dict)
            for (sub, ses) in self.sub_ses
        }
        avail_seeds = [
            int(x.name.replace("subclasses_", ""))
            for x in self.seed_path.glob("subclasses_*")
        ]
        min_seeds_available = min(avail_seeds)
        max_seeds_available = max(avail_seeds)
        for n_sub in range(
            min_seeds_available,
            max_seeds_available + 1,
        ):
            seed_path = self.seed_path / f"subclasses_{n_sub}"
            if not seed_path.exists():
                raise FileNotFoundError(
                    f"Provided seed path {seed_path} does not exist."
                )
            # load the seeds for the subjects for each meta label 1-4
            for i in range(1, 5):
                files = self._load_metalabel_path(seed_path, f"mlabel_{i}")
                for (sub, ses), file in zip(self.sub_ses, files):
                    sub_ses_str = self._sub_ses_string(sub, ses)
                    self.seed_paths[sub_ses_str][n_sub][i] = file

    def _load_seed_csf_path(self):
        """Load the seeds for the subjects."""
        self.seed_csf_paths = {
            self._sub_ses_string(sub, ses): defaultdict(dict)
            for (sub, ses) in self.sub_ses
        }
        avail_seeds = [
            int(x.name.replace("subclasses_", ""))
            for x in self.seed_csf_path.glob("subclasses_*")
        ]
        min_seeds_available = min(avail_seeds)
        max_seeds_available = max(avail_seeds)
        for n_sub in range(
            min_seeds_available,
            max_seeds_available + 1,
        ):
            seed_csf_path = self.seed_csf_path / f"subclasses_{n_sub}"
            if not seed_csf_path.exists():
                raise FileNotFoundError(
                    f"Provided seed csf path {seed_csf_path} does not exist."
                )
            # load the seeds for the subjects for each meta label 1-4
            for i in range(1, 5):
                files = self._load_metalabel_path(seed_csf_path, f"mlabel_{i}")
                for (sub, ses), file in zip(self.sub_ses, files):
                    sub_ses_str = self._sub_ses_string(sub, ses)
                    self.seed_csf_paths[sub_ses_str][n_sub][i] = file


    def sample(self, idx, genparams: dict = {}) -> tuple[dict, dict]:
        """
        Retrieve a single item from the dataset at the specified index.

        Args:
            idx (int): The index of the item to retrieve.
            genparams (dict): Dictionary with generation parameters.
                Used for fixed generation. Should follow exactly the same structure
                and be of the same type as the returned generation parameters.
                Can be used to replicate the augmentations (power)
                used for the generation of a specific sample.
        Returns:
            Dictionaries with the generated data and the generation parameters.
                First dictionary contains the `image`, `label` and the `name` keys.
                The second dictionary contains the parameters used for the generation.

        !!! Note
            The `image` is scaled to `[0, 1]` and oriented with the `label` to **RAS**
            and returned on the device  specified in the `generator` initialization.
        """
        # use generation_params to track the parameters used for the generation
        generation_params = {}
        image = self.loader(self.img_paths[idx]) if self.load_image else None
        segm = self.loader(self.segm_paths[idx])

        # orient to RAS for consistency
        image = (
            self.orientation(image.unsqueeze(0)).squeeze(0) if self.load_image else None
        )
        segm = self.orientation(segm.unsqueeze(0)).squeeze(0)

        # transform name into a single string otherwise collate fails
        name = self.sub_ses[idx]
        name = self._sub_ses_string(name[0], ses=name[1])

        # initialize seeds as dictionary
        # with paths to the seeds volumes
        # or None if image is to be used as intensity prior
        if self.seed_path is not None:
            seeds = self.seed_paths[name]
        if self.seed_csf_path is not None:
            seeds_csf = self.seed_csf_paths[name]
        if self.image_as_intensity:
            seeds = None
            seeds_csf = None

        # log input data
        generation_params["idx"] = idx
        generation_params["img_paths"] = str(self.img_paths[idx])
        generation_params["segm_paths"] = str(self.img_paths[idx])
        generation_params["seeds"] = str(self.seed_path)
        generation_params["seeds_csf"] = str(self.seed_csf_path)
        generation_time_start = time.time()

        # generate the synthetic data
        gen_output, segmentation, image, synth_params = self.generator.sample(
            image=image, segmentation=segm, seeds=seeds, seeds_csf=seeds_csf, genparams=genparams
        )

        # scale the images to [0, 1]
        gen_output = self.scaler(gen_output)
        image = self.scaler(image) if image is not None else None

        # ensure image and segmentation are on the cpu
        gen_output = gen_output.cpu()
        segmentation = segmentation.cpu()
        image = image.cpu() if image is not None else None

        generation_params = {**generation_params, **synth_params}
        generation_params["generation_time"] = time.time() - generation_time_start
        data_out = {
            "image": gen_output.unsqueeze(0),
            "label": segmentation.unsqueeze(0).long(),
            "name": name,
        }

        return data_out, generation_params

    def __getitem__(self, idx) -> dict:
        """
        Retrieve a single item from the dataset at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Dictionary with the `image`, `label` and the `name` keys.
                `image` and `label` are `torch.float32`
                [`monai.data.meta_tensor.MetaTensor`](https://docs.monai.io/en/stable/data.html#metatensor)
                and `name` is a string of a format `sub_ses` where `sub` is the subject name
                and `ses` is the session name.

        !!!Note
            The `image` is scaled to `[0, 1]` and oriented to **RAS** and returned on the device
            specified in the `generator` initialization.
        """
        data_out, generation_params = self.sample(idx)
        self.generation_params = generation_params
        return data_out

    def sample_with_meta(self, idx: int, genparams: dict = {}) -> dict:
        """
        Retrieve a sample along with its generation parameters
        and store them in the same dictionary.

        Args:
            idx: The index of the sample to retrieve.
            genparams: Dictionary with generation parameters.
                Used for fixed generation. Should follow exactly the same structure
                and be of the same type as the returned generation parameters from the `sample()` method.
                Can be used to replicate the augmentations (power)
                used for the generation of a specific sample.

        Returns:
            A dictionary with `image`, `label`, `name` and `generation_params` keys.
        """

        data, generation_params = self.sample(idx, genparams=genparams)
        data["generation_params"] = generation_params
        return data
    