import hydra
from omegaconf import DictConfig, OmegaConf
import os
import nibabel as nib
import json


@hydra.main(version_base=None, config_path="./../configs", config_name="test")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # instantiate the generator
    cfg = hydra.utils.instantiate(cfg)
    # print(cfg.dataset.bids_path)
    print(len(cfg.dataset))

    for i in range(0, 100, 5):
        data = cfg.dataset[i]
        metadata = cfg.dataset.generation_params
        # cfg.dataset.reverse_transform(data)
        # print(cfg.dataset.sample(1)[1])
        print(data["image"].shape)
        print(data["label"].shape)
        print(data["name"])
        # types
        print(type(data["image"]))
        print(data["image"].dtype)
        print(type(data["label"]))
        print(data["label"].dtype)

        print(data["image"].device)
        print(data["label"].device)
        print(data["image"].max())
        print(data["label"].max())
        print(data["image"].min())
        print(data["label"].min())
        os.makedirs("test", exist_ok=True)
        nib.save(
            nib.Nifti1Image(data["image"].cpu().numpy(), affine=None),
            f"test/image_{i}.nii.gz",
        )
        print(metadata)
        with open(f"test/image_{i}.json", "w") as f:
            json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    my_app()
