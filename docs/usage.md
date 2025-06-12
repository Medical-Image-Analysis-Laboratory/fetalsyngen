# How to use the generator?

Follow these steps to integrate and use the generator in your project:

## 1. Install the Package
Refer to the [Installation](index.md#installation) page for detailed instructions on how to install the package.

## 2. Prepare the Dataset
Ensure your dataset is formatted according to the [BIDS format](https://bids.neuroimaging.io/). Your dataset must include the following files:

- **T2w image**: Files should have the naming pattern `*_T2w.nii.gz`.
- **Segmentation mask**: Files should have the naming pattern `*_dseg.nii.gz`.
- **Seeds**: Seeds used for subclass generation. It is recommended to store them in a separate folder within the BIDS dataset, such as `derivatives/seeds`.

!!! Note
    > **📝 Seeds**: The **seeds must be generated** using the `generate_seeds.py` script provided in the package. See the [seed generation page](seed_generation.md) for more details.
    
    > **📝 Resampling**: For the correct work of the generator, all input data needs to be resampled to the same spatial size and resolution. Please perform this step before seed generation.

## 3. Copy Dataset Configurations

We provide a variety of ready-to-use configurations for different tasks. These configuration files are stored in the [`fetalsyngen/configs/dataset`](https://github.com/Medical-Image-Analysis-Laboratory/fetalsyngen/tree/dev/configs/dataset) folder and are further detailed in the [Configs](configs.md) page.

Each configuration is a `.yaml` file that contains the parameters for the generation pipeline. You can modify these configurations to suit the specific requirements of your project.

## 4. Run the Generator

We offer several `torch.Dataset` classes for loading synthetic and real datasets:

- **`fetalsyngen.data.datasets.FetalTestDataset`**: Loads real images and segmentations. Used for testing and validation on real data.
- **`fetalsyngen.data.datasets.FetalSynthDataset`**: Can be used to either to create synthetic images and segmentation on the fly or apply the same transformations used in generation of synthetic data to real images and segmentations.

For more details on these datasets, see the [Datasets](datasets.md) page.

!!! Note
    > **📝 Configs**: Use the local copy of the config files from your repository to instantiate the generator/dataset classes.

    > **📝 Paths**: When using the dataset classes, ensure that the paths in your local configuration files are updated to correctly reference your dataset and seed files.

