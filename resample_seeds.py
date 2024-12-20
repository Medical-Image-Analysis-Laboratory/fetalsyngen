import nibabel as nib
from pathlib import Path
from monai.transforms import LoadImage
import SimpleITK as sitk
from tqdm import tqdm

PATH = Path(
    "/media/vzalevskyi/data/FETA_challenge/merged_feta_spinabifida/derivatives/seeds"
)
igms = list(PATH.glob("**/*.nii.gz"))
for imgp in tqdm(igms):
    img = nib.load(str(imgp))
    img_header = img.header
    img_header.set_data_dtype("int8")
    img_int8 = nib.Nifti1Image(img.get_fdata().astype("int8"), img.affine, img.header)
    nib.save(img_int8, str(imgp))
