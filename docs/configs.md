# Configs

We use Hydra to manage configurations and instantiate classes in the `FetalSynthGen` pipeline. It allows us to define configurations in YAML files and instantiate classes with these configurations. This makes it easy to modify parameters and experiment with different settings.

See the [Hydra documentation](https://hydra.cc/docs/intro) for more information.

## Quick hydra overview

Configuration files are stored in the [`configs`](https://github.com/Medical-Image-Analysis-Laboratory/fetalsyngen/tree/dev/configs) directory.

Each file is a `.yaml` file that contains the parameters for the generation pipeline and defines an instantiation of a class.

Fields in the configuration files can be overridden from the command line or from other configuration files. Some special fields are:

* `_target_` field in the configuration file specifies the class to be instantiated.
All other fields are passed as arguments to the class constructor. It can be any callable object, including classes, functions, and lambdas, from any module in the Python path (local or external).
* `defaults` field in the configuration file specifies the default configuration file to be used, that is merged with the current configuration file. If a field is present in both files, the one in the current file takes precedence.

<blockquote>
For example

```
defaults:
    - generator/default
```
</blockquote>

> will load the `generator/default.yaml` file and will make all fields from the `generator/default.yaml` file available in the current configuration file from the `generator.*` namespace.


* `null` is used in the configuration files to specify a `None` value in Python.
* expressions like `"${var}"` can be used to access variable value in the same level of a given config while `"${..var}"` can be used to access variable value from the parent config.

## Configuration Files
We provide a variety of ready-to-use configurations for different tasks. These configuration files are stored in the [`fetalsyngen/configs/dataset`](https://github.com/Medical-Image-Analysis-Laboratory/fetalsyngen/tree/dev/configs) directory.

To use them, copy the configuration files to your project root directory into `configs/dataset`. Feel free to modify these configurations to suit the specific requirements of your project.


## Validation/Testing Dataset
Dataset configuration for loading real images and segmentations. Used for testing and validation on real data. See [`/datasets/#fetalsyngen.data.datasets.FetalTestDataset`](datasets.md#fetalsyngen.data.datasets.FetalTestDataset) for more details.
```yaml
defaults:
  - transforms/inference

_target_: fetalsyngen.data.datasets.FetalTestDataset
bids_path: ./data
sub_list: null
```

## Real images with synthetic transformations
Dataset configuration for applying the same transformations used in the generation of synthetic data to real images and segmentations. See [`/datasets/#fetalsyngen.data.datasets.FetalSynthDataset`](datasets.md#fetalsyngen.data.datasets.FetalSynthDataset) for more details.

`configs/dataset/real_train.yaml` >
```yaml
defaults:
  - generator/default

_target_: fetalsyngen.data.datasets.FetalSynthDataset
bids_path: ./data
seed_path: null
sub_list: null
load_image: True
image_as_intensity: True
```

## Synthetic images and segmentations
Dataset configuration for creating synthetic images and segmentations on the fly. See [`/datasets/#fetalsyngen.data.datasets.FetalSynthDataset`](datasets.md#fetalsyngen.data.datasets.FetalSynthDataset) for more details.

`configs/dataset/synth_train.yaml` >
```yaml
defaults:
  - generator/default

_target_: fetalsyngen.data.datasets.FetalSynthDataset
bids_path: ./data
seed_path: ./data/derivatives/seeds
sub_list: null
load_image: False
image_as_intensity: False
```

## Default Generator Configuration
`configs/dataset/generator/default.yaml` >
```yaml
_target_: fetalsyngen.generator.model.FetalSynthGen

shape: [256, 256, 256]
resolution: [0.5, 0.5, 0.5]
device: cuda # cuda ~6x faster than cpu

intensity_generator:
  _target_: fetalsyngen.generator.intensity.rand_gmm.ImageFromSeeds
  min_subclusters: 1
  max_subclusters: 3
  seed_labels: [0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
  generation_classes: [0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

spatial_deform:
  _target_: fetalsyngen.generator.deformation.affine_nonrigid.SpatialDeformation
  device: "${..device}" # uses the device from the generator
  size: ${..shape} # uses the shape from the generator
  flip_prb: 0.5

  max_rotation: 20
  max_shear: 0.02
  max_scaling: 0.1

  nonlinear_transform: True
  nonlin_scale_min: 0.03
  nonlin_scale_max: 0.06
  nonlin_std_max: 4

resampler:
  _target_: fetalsyngen.generator.augmentation.synthseg.RandResample
  min_resolution: 1.9
  max_resolution: 2
  prob: 1

bias_field:
  _target_: fetalsyngen.generator.augmentation.synthseg.RandBiasField
  prob: 1
  scale_min: 0.004 
  scale_max: 0.02
  std_min:  0.01
  std_max: 0.3

gamma:
  _target_: fetalsyngen.generator.augmentation.synthseg.RandGamma
  prob: 1
  gamma_std: 0.1

noise:
  _target_: fetalsyngen.generator.augmentation.synthseg.RandNoise
  prob: 1
  std_min: 5
  std_max: 15
```