# Examples
After installing the package, you can directly use the generator and datasets in your project. See the following examples for guidance on how to instantiate the generator and datasets.

## Recommended: Using Configuration Files

For reproducibility and greater flexibility, we recommend using the configuration files provided in the package. These files define the parameters for the generator and datasets, allowing for quick and easy setup with [`hydra`](https://hydra.cc/).

### Steps to Use Configuration Files

1. Copy the configuration files (entire [`configs/dataset`](https://github.com/Medical-Image-Analysis-Laboratory/fetalsyngen/tree/dev/configs/dataset) folder) to your project root directory into `configs/dataset`.
2. Use the following methods to instantiate the generator and dataset classes:


For examples below, set up `cfg_path = "configs/dataset"` as the path to the configuration files and `cfg_name` as the name of the configuration file you want to use (`cfg_name='synth_train'` for example for the synthetic training dataset).

See the [`Configs`](configs.md) page for detailed information on configuration files and available generation modes.

<details open>

Using the Imperative API

```python
import hydra

with hydra.initialize(config_path=cfg_path, version_base="1.2"):
    cfg = hydra.compose(config_name=cfg_name)
    print(f"Composed config: {cfg}")
    dataset = hydra.utils.instantiate(cfg)
```

Using the Declarative API

```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path=cfg_path, config_name=cfg_name)
def my_app(cfg: DictConfig) -> None:
    print(cfg)
```
</details>
<br>
**Note:** Ensure that the `bids` and `seeds` paths in the configuration files are updated to the absolute paths for your data.


## Using Direct Instantiation


You can manually instantiate required classes from the `FetalSynthGen` in your project as needed. For example, to instantiate the `FetalSynthDataset` class and generator components follow the example below:

<details close>

```python
# Import necessary classes
from fetalsyngen.data.datasets import FetalSynthDataset

from fetalsyngen.generator.model import FetalSynthGen
from fetalsyngen.generator.augmentation.synthseg import (
    RandBiasField,
    RandGamma,
    RandNoise,
    RandResample,
)
from fetalsyngen.generator.deformation.affine_nonrigid import SpatialDeformation
from fetalsyngen.generator.intensity.rand_gmm import ImageFromSeeds

# Instantiate the generator components
intensity_generator = ImageFromSeeds(
    min_subclusters=1,
    max_subclusters=3,
    seed_labels=[1, 2, 3, 4, 5],
    generation_classes=[1, 2, 3, 4, 5],
    meta_labels=4,
)
spatial_deform = SpatialDeformation(
    max_rotation=10,
    max_shear=1,
    max_scaling=1,
    size=(256, 256, 256),
    nonlinear_transform=1,
    nonlin_scale_min=1,
    nonlin_scale_max=1,
    nonlin_std_max=1,
    flip_prb=1,
    device="cuda",
)
resampler = RandResample(prob=0.5, max_resolution=1.5, min_resolution=0.5)
bias_field = RandBiasField(
    prob=0.5, scale_min=0.5, scale_max=1.5, std_min=0.5, std_max=1.5
)
noise = RandNoise(prob=0.5, std_min=0.5, std_max=1.5)
gamma = RandGamma(prob=0.5, gamma_std=0.5)

# Instantiate the generator
generator = FetalSynthGen(
    shape=(256, 256, 256),
    resolution=(0.5, 0.5, 0.5),
    device="cuda",
    intensity_generator=intensity_generator,
    spatial_deform=spatial_deform,
    resampler=resampler,
    bias_field=bias_field,
    noise=noise,
    gamma=gamma,
)

# Instantiate the dataset
dataset = FetalSynthDataset(
    bids_path="./../../data",
    generator=generator,
    seed_path="./../../data/derivatives/seeds",
    sub_list=None,
)
```

</details>

<br>

## Additional Resources

For more examples of generator instantiation with `hydra`, see the [`Generator Instantiation`](https://github.com/Medical-Image-Analysis-Laboratory/fetalsyngen/blob/dev/fetalsyngen/examples/generator.ipynb) notebook.

