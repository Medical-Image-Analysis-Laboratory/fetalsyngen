# Datasets
Classes for loading and processing datasets.
### Note
> **📝 Device**: All datasets return samples with tensors on the CPU (even when the synthetic data generation is done on the GPU). This is due to restriction on the GPU usage in the multiprocessing settings, where GPU memory cannot be easily shared between processes.

> **📝 Dataloader**: When using `torch.utils.data.DataLoader` ensure that you pass `multiprocessing_context="spawn"` argument to the dataloader object when using `FetalSynthDataset` to ensure that the spawned processes have access to the GPU.


::: fetalsyngen.data.datasets