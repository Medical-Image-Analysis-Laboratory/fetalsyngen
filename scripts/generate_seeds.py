"""Run this scripts to generate seeds used for FetalSynthGen.

Uses segmentation label -> meta-label mapping defined at `tissue_map` to fuse
similar classes into the same label.

Then splits it into N clusters using EM clustering.

All non-zero voxels from the image that are background in the segmentation
are clustered into 4th clusters (MAX_BACK_SUBCLUST) also defines how mby subclasses
to simulate for the background tissue).
"""

from pathlib import Path
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import monai
import torch
import argparse
import numpy as np
from multiprocessing import Pool


parser = argparse.ArgumentParser(
    description="Generate seeds for FetalSynthGen",
    epilog="Example: python scripts/generate_seeds.py --bids_path /path/to/bids --out_path /path/to/out --max_subclasses 6 --annotation feta",
)

parser.add_argument(
    "--bids_path",
    type=str,
    required=True,
    help="Path to BIDS folder with the segmentations and images for seeds generation",
)
parser.add_argument("--out_path", type=str, required=True, help="Path to save the seeds")
parser.add_argument(
    "--max_subclasses",
    type=int,
    default=10,
    help="How many subclasses to simulate for each tissue type (meta-label)",
)
parser.add_argument(
    "--annotation",
    type=str,
    required=True,
    help="Annotation type. Should be either 'feta' or 'dhcp'",
    choices=["feta", "dhcp"],
)
args = parser.parse_args()


def main(args):
    # mapping fetal labels to meta labels
    if args.annotation == "feta":
        tissue_map = {
            "CSF": [1, 4],  # meta-label 1
            "GM": [2, 6],  # meta-label 2
            "WM": [3, 5, 7],  # meta-label 3
        }
        feta2meta = {1: 1, 4: 1, 2: 2, 6: 2, 5: 3, 7: 3, 3: 3}
    elif args.annotation == "dhcp":
        tissue_map = {
            "CSF": [1, 5],  # meta-label 1
            "GM": [2, 7, 9],  # meta-label 2
            "WM": [3, 6, 8],  # meta-label 3
        }
        feta2meta = {1: 1, 5: 1, 2: 2, 7: 2, 9: 2, 3: 3, 6: 3, 8: 3}
    else:
        raise ValueError("Unknown annotation type. Should be either 'feta' or 'dhcp'")

    print(f'Using "{args.annotation}" annotation. Labels are mapped as follows:')
    for meta_label, segm_labels in tissue_map.items():
        print(f"Meta-label {meta_label} is a fusion of segmentation labels: {segm_labels}")

    max_subclusts = int(args.max_subclasses) + 1
    bids_path = Path(args.bids_path).absolute()
    out_path = Path(args.out_path).absolute()
    loader = monai.transforms.LoadImaged(keys=["image", "label"])
    subjects = list(bids_path.glob("sub-*"))

    print(f"Found {len(subjects)} subjects in {bids_path}")

    # Prepare input arguments for parallel processing
    tasks = []
    for sub in subjects:
        for subclasses in range(1, max_subclusts):
            imgs = list(sub.glob("**/anat/*_T2w.nii.gz"))[0]
            label = list(sub.glob("**/anat/*_dseg.nii.gz"))[0]
            tasks.append((imgs, label, subclasses, feta2meta, out_path, sub, "", loader))

    # Use multiprocessing Pool for parallel processing
    with Pool(mp.cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(worker_process_subject, tasks), total=len(tasks)))


def worker_process_subject(args):
    """Wrapper for process_subject to unpack arguments for multiprocessing."""
    process_subject(*args)


def process_subject(imgs, label, subclasses, feta2meta, out_path, sub, session, loader):
    data = loader({"image": str(imgs), "label": str(label)})
    data["image"] = data["image"].unsqueeze(0)
    data["label"] = data["label"].unsqueeze(0)
    # set skull as class 4
    data["label"][data["label"] == 4] = 0

    subclasses_splits = split_lables(
        image=data["image"],
        segmentation=data["label"],
        subclasses=subclasses,
        feta2meta=feta2meta,
    )
    for n_subclasses, subsegms in subclasses_splits.items():
        for mlabel, subsegm in subsegms.items():
            # different output formats for when session is present or not
            if session == "":
                out_suffix = f"subclasses_{n_subclasses}/{sub.name}/anat/"
            else:
                out_suffix = f"subclasses_{n_subclasses}/{sub.name}/{session}/anat/"

            # save the subsegmentations with int8 dtype
            saver_segm = monai.transforms.SaveImaged(
                keys=["label"],
                output_dir=out_path / out_suffix,
                output_postfix=f"mlabel_{mlabel}",
                resample=False,
                separate_folder=False,
                output_dtype=np.int8,
                print_log=False,
                allow_missing_keys=True,
                mode="nearest",
            )

            subclas_data = {"label": subsegm}
            saver_segm(subclas_data)


def subsplit_label(img, segm, label2assign=10, n_clusters=3):
    img_voxels = img[segm > 0]
    # cluster non-zero image voxels that are zero in the mask
    brain_backg = segm * 0

    clust = GaussianMixture(n_components=n_clusters, n_init=5, init_params="k-means++").fit_predict(
        img_voxels.reshape(-1, 1)
    )
    clust = torch.tensor(clust).long()
    brain_backg[segm > 0] = clust + label2assign  # clusters are from 0 to n_clusters-1
    return brain_backg


def split_lables(image, segmentation, subclasses, feta2meta):
    # fuse feta labels into meta labels
    meta_segm = segmentation * 0
    for fetalab, metalab in feta2meta.items():
        meta_segm[segmentation == fetalab] = metalab

    # set skull as class 4
    meta_segm[(segmentation == 0) & (image != 0)] = 4
    sublclasses = {}  # dictionary to store the subsegmentations
    # in a format { number_of_subclasses: {meta_labels: subclasses_mask} }
    if subclasses == 1:
        sublclasses[subclasses] = {x: (meta_segm == x) * x * 10 for x in range(1, 5, 1)}
        return sublclasses
    else:
        sublclasses[subclasses] = {}
        for metalabel in range(1, 5):
            mlabel_mask = meta_segm == metalabel
            split_segm = subsplit_label(
                image, mlabel_mask, label2assign=10 * metalabel, n_clusters=subclasses
            )
            sublclasses[subclasses][metalabel] = split_segm
        return sublclasses


if __name__ == "__main__":
    main(args)
