from collections import defaultdict
from pathlib import Path
from monai.transforms import (
    LoadImage,
    ScaleIntensityd,
    Orientationd,
)
from monai.transforms import Compose
from fetalsyngen.generator import SynthGen
from hydra.utils import instantiate
from fetalsyngen.dataset.readers import SimpleITKReader
import time


# TODO: keep in mind base_transforms (croppings) to be applied and think of the way to aooly them
# TODO: Keep in mind scaling and orientation in getitem of all datasets
class FetalDataset:
    """Abstract class defining a dataset for loading fetal data."""

    def __init__(
        self,
        bids_path: str,
        sub_list: list[str] | None,
    ):
        """
        Args:

            bids_path: Path to the bids folder with the data.
            sub_list: List of the subjects to use.

        Returns:

            None
        """
        super().__init__()

        self.bids_path = Path(bids_path)
        self.subjects = sub_list
        if self.subjects is None:
            self.subjects = [x.name for x in self.bids_path.glob("sub-*")]
        self.sub_ses = [
            (x, y) for x in self.subjects for y in self._get_ses(self.bids_path, x)
        ]
        self.loader = SimpleITKReader()
        self.scaler = ScaleIntensityd(
            keys=["image"],
            minv=0,
            maxv=1,
        )

        self.orientation = Orientationd(
            axcodes="RAS", keys=["image", "label"], allow_missing_keys=True
        )

        self.img_paths = self._load_bids_path(self.bids_path, "T2w")
        self.segm_paths = self._load_bids_path(self.bids_path, "dseg")

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

    def _load_bids_path(self, path, suffix):
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

        return files_paths

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        raise NotImplementedError(
            "This method should be implemented in the child class."
        )


class FetalSimpleDataset(FetalDataset):
    """Dataset class for loading real fetal.
    Used to load test/validation data.
    By default it loads the image and the segmentation,
    scales the intensities to [0, 1] and orients the data to RAS.

    Pass the `transforms` for additional processing
    (scaling, resampling, cropping, etc.) as an argument.
    """

    def __init__(
        self,
        bids_path: str,
        sub_list: list[str] | None,
        transforms: Compose | None = None,
    ):
        """

        Args:

            bids_path: Path to the bids folder with the data.
            sub_list: List of the subjects to use.
            transforms: Compose object with the transformations to apply.
                Default is None, no transformations are applied.
        """
        super().__init__(bids_path, sub_list)
        self.transforms = transforms

    def __getitem__(self, idx):
        """Get the data for the given index.

        Returns:
            dict: Dictionary with the `image`, `label` and the `name`
                keys. `image` and `label` are monai meta tensors
                with dimensions (1, x, y, z) and `name` is a string
                of a format `sub_ses` where `sub` is the subject name
                and `ses` is the session name.
        """
        image = self.loader(self.img_paths[idx])
        segm = self.loader(self.segm_paths[idx])

        image = image.unsqueeze(0)  # can be fixed by ensure_channel_first? TODO
        segm = segm.unsqueeze(0)

        # transform name into a single string otherwise collate fails
        name = self.sub_ses[idx]
        name = self._sub_ses_string(name[0], ses=name[1])

        data = {"image": image, "label": segm, "name": name}

        # orient to RAS for consistency
        data = self.orientation(data)

        if self.transforms:
            data = self.transforms(data)

        data = self.scaler(data)

        return data

    def reverse_transform(self, data):
        """Reverse the transformations applied to the data.

        Args:
            data: Dictionary with the `image` and `label` keys.

        Returns:
            dict: Dictionary with the `image` and `label` keys.
        """
        if self.transforms:
            data = self.transforms.inverse(data)
        data = self.orientation.inverse(data)
        return data


class FetalSynthDataset(FetalDataset):
    """Dataset class for generating
    synthetic fetal data"
    """

    def __init__(
        self,
        bids_path: str,
        generator: SynthGen,
        seed_path: str | None,
        sub_list: list[str] | None,
        load_image: bool = False,
    ):
        """

        Args:

            bids_path: Path to the bids folder with the data.
            seed_path: Path to the folder with the seeds to use for
                intensity sampling. See `scripts/seed_generation.py`
                for details on the data formatting. If seed_path is None,
                the intensity  sampling step is skipped and the output image
                intensities will be based on the input image.
            generator: SynthGen object with the generator to use.
            sub_list: List of the subjects to use.
            load_image: If True, the image is loaded and passed to the generator,
                where it can be used as the intensity prior instead of a random
                intensity sampling or spatially deformed with the same transformation
                field as segmentation and the syntehtic image. Default is False.
        """
        super().__init__(bids_path, sub_list)
        self.seed_path = Path(seed_path) if isinstance(seed_path, str) else None
        self.load_image = load_image
        self.generator = generator

        # parse seeds paths
        if isinstance(self.seed_path, Path):
            if not self.seed_path.exists():
                raise FileNotFoundError(
                    f"Provided seed path {self.seed_path} does not exist."
                )
            else:
                self._load_seed_path()

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
                files = self._load_bids_path(seed_path, f"mlabel_{i}")
                for (sub, ses), file in zip(self.sub_ses, files):
                    sub_ses_str = self._sub_ses_string(sub, ses)
                    self.seed_paths[sub_ses_str][n_sub][i] = file

    def sample(self, idx):
        """Get the data for the given index.

        Returns:
            dict: Dictionary with the `image`, `label` and the `name`
                keys. `image` and `label` are monai meta tensors
                with dimensions (1, x, y, z) and `name` is a string
                of a format `sub_ses` where `sub` is the subject name
                and `ses` is the session name.
        """
        # use generation_params to track the parameters used for the generation
        generation_params = {}

        image = self.loader(self.img_paths[idx]) if self.load_image else None
        segm = self.loader(self.segm_paths[idx])

        # transform name into a single string otherwise collate fails
        name = self.sub_ses[idx]
        name = self._sub_ses_string(name[0], ses=name[1])

        # initialize seeds as dictionary
        # with paths to the seeds volumes
        # or None if image is to be used as intensity prior
        if self.seed_path is not None:
            seeds = self.seed_paths[name]
        else:
            seeds = None

        # log input data
        generation_params["idx"] = idx
        generation_params["img_paths"] = self.img_paths[idx]
        generation_params["segm_paths"] = self.img_paths[idx]
        generation_params["seeds"] = seeds
        generation_time_start = time.time()

        # generate the synthetic data
        gen_output, segmentation, image, synth_params = self.generator.sample(
            image=image, segmentation=segm, seeds=seeds
        )

        generation_params = {**generation_params, **synth_params}
        generation_params["generation_time"] = time.time() - generation_time_start
        return {
            "image": gen_output,
            "label": segmentation,
            "name": name,
        }, generation_params

    def __getitem__(self, idx):
        return self.sample(idx)[0]

    def sample_with_meta(self, idx):
        data, generation_params = self.sample(idx)
        data["generation_params"] = generation_params
        return data
