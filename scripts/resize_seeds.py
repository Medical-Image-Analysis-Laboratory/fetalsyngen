"""Casts the seeds to int8"""

import nibabel as nib
from pathlib import Path
from monai.transforms import LoadImage
import SimpleITK as sitk
from tqdm import tqdm
import argparse


def main(path):
    PATH = Path(path)
    igms = list(PATH.glob("**/*.nii.gz"))
    for imgp in tqdm(igms):
        img = nib.load(str(imgp))
        img_header = img.header
        img_header.set_data_dtype("int8")
        img_int8 = nib.Nifti1Image(
            img.get_fdata().astype("int8"), img.affine, img.header
        )
        nib.save(img_int8, str(imgp))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cast the seeds to int8")
    parser.add_argument(
        "path",
        type=str,
        help="Path to the directory containing the seed files. Example: /path/to/derivatives/seeds",
    )
    args = parser.parse_args()
    main(args.path)
