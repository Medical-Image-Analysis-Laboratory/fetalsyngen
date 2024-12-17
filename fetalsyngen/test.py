import hydra
from omegaconf import DictConfig, OmegaConf


# TODO: Have a config loading that doesn't depent on decorators of hydra
# and give examples of, so it can be flexibly run from
# any point in the the computer

# 2.
# TODO: Write a dataloader to see if spawning multiple
# proceess and see if metatensors are causing an issue

# TODO: Explain the sample and get_item difference


@hydra.main(version_base=None, config_path="./../configs", config_name="fetalsynthgen")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # instantiate the generator
    cfg = hydra.utils.instantiate(cfg)
    # print(cfg.dataset.bids_path)
    print(len(cfg.dataset))

    data = cfg.dataset[0]
    print(cfg.dataset.sample(1)[1])
    # print(data)
    # print(
    #     f"Image - shape: {data['image'].shape}, dtype: {data['image'].dtype}, max: {data['image'].max()}, min: {data['image'].min()}, device: {data['image'].device}, type: {type(data['image'])}"
    # )
    # print(
    #     f"Label - shape: {data['label'].shape}, dtype: {data['label'].dtype}, max: {data['label'].max()}, min: {data['label'].min()}, device: {data['label'].device}, type: {type(data['label'])}"
    # )

    # data = cfg.dataset.reverse_transform({"label": data["label"]})
    # print(
    #     f"Label - shape: {data['label'].shape}, dtype: {data['label'].dtype}, max: {data['label'].max()}, min: {data['label'].min()}, device: {data['label'].device}, type: {type(data['label'])}"
    # )


if __name__ == "__main__":
    my_app()
