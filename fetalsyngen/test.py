import hydra
from omegaconf import DictConfig, OmegaConf


# TODO: Have a config loading that doesn't depent on decorators of hydra
# and give examples of, so it can be flexibly run from
# any point in the the computer

# 2.
# TODO: Write a dataloader to see if spawning multiple
# proceess and see if metatensors are causing an issue

# TODO: Explain the sample and get_item difference


@hydra.main(version_base=None, config_path="./../configs", config_name="test")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # instantiate the generator
    cfg = hydra.utils.instantiate(cfg)
    # print(cfg.dataset.bids_path)
    print(len(cfg.dataset))

    data = cfg.dataset[0]
    # print(cfg.dataset.sample(1)[1])
    print(data["image"].shape)
    print(data["label"].shape)
    print(data["name"])
    # types
    print(type(data["image"]))
    print(data["image"].dtype)
    print(type(data["label"]))
    print(data["label"].dtype)


if __name__ == "__main__":
    my_app()
