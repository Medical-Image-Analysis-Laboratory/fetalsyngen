import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import time


@hydra.main(version_base=None, config_path="./../configs", config_name="test")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # instantiate the generator
    cfg = hydra.utils.instantiate(cfg)
    # print(cfg.dataset.bids_path)
    dl = DataLoader(
        cfg.dataset,
        batch_size=2,
        num_workers=2,
        multiprocessing_context="spawn",
        # pin_memory=True,
    )

    # evaluate the speed of the dataloader
    start = time.time()
    for data in tqdm(dl):
        pass
    end = time.time()
    print(f"Time taken for dataloader: {end-start:.2f} seconds")


if __name__ == "__main__":
    my_app()
