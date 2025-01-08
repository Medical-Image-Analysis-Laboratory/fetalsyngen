# Datasets
Classes for loading and processing datasets.
### Note
> **📝 Device**: All datasets return samples with tensors on the CPU (even when the synthetic data generation is done on the GPU). This is due to restriction on the GPU usage in the multiprocessing settings, where GPU memory cannot be easily shared between processes.

> **📝 Dataloader**: When using `torch.utils.data.DataLoader` ensure that you pass `multiprocessing_context="spawn"` argument to the dataloader object when using `FetalSynthDataset` to ensure that the spawned processes have access to the GPU.


::: fetalsyngen.data.datasets

## Fixed Image Generation
It is possible to generate synthetic images of the same 'augmentation' power as any given synthetic image. This is done by passing the `genparams` dictionary to the [`sample_with_meta`](#fetalsyngen.data.datasets.FetalSynthDataset.sample_with_meta) (or [`sample`](#fetalsyngen.data.datasets.FetalSynthDataset.sample)) method of the [`FetalSynthDataset`](#fetalsyngen.data.datasets.FetalSynthDataset) class. The `generation_params` dictionary is a dictionary of the parameters used to generate the image. The method will then use these parameters to generate a new image with the same `augmentation power` as the original image.

This `genparams` dictionary can be obtained, for example, from the dictionary returned by the [`FetalSynthDataset.sample_with_meta`](#fetalsyngen.data.datasets.FetalSynthDataset.sample_with_meta)  method. It then can be directly used to `fix` (some or all) generation parameters for the new image.

See example below:

```python
# initialize the dataset class
# see the Examples page for more details
dataset = FetalSynthDataset(...)

# first sample a synthetic image from the dataset
sample = dataset.sample_with_meta(0)
# then we sample a synthetic image with the same augmentation power as the first image
sample_copy = dataset.sample_with_meta(0, genparams=sample["generation_params"])
```

For example, generation parameters of the first image can be like this:

<details>
```python
{'idx': 0,
 'img_paths': PosixPath('../data/sub-sta38/anat/sub-sta38_rec-irtk_T2w.nii.gz'),
 'segm_paths': PosixPath('../data/sub-sta38/anat/sub-sta38_rec-irtk_T2w.nii.gz'),
 'seeds': defaultdict(dict,
             {1: {1: PosixPath('../data/derivatives/seeds/subclasses_1/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_1.nii.gz'),
               2: PosixPath('../data/derivatives/seeds/subclasses_1/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_2.nii.gz'),
               3: PosixPath('../data/derivatives/seeds/subclasses_1/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_3.nii.gz'),
               4: PosixPath('../data/derivatives/seeds/subclasses_1/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_4.nii.gz')},
              2: {1: PosixPath('../data/derivatives/seeds/subclasses_2/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_1.nii.gz'),
               2: PosixPath('../data/derivatives/seeds/subclasses_2/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_2.nii.gz'),
               3: PosixPath('../data/derivatives/seeds/subclasses_2/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_3.nii.gz'),
               4: PosixPath('../data/derivatives/seeds/subclasses_2/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_4.nii.gz')},
              3: {1: PosixPath('../data/derivatives/seeds/subclasses_3/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_1.nii.gz'),
               2: PosixPath('../data/derivatives/seeds/subclasses_3/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_2.nii.gz'),
               3: PosixPath('../data/derivatives/seeds/subclasses_3/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_3.nii.gz'),
               4: PosixPath('../data/derivatives/seeds/subclasses_3/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_4.nii.gz')}}),
 'selected_seeds': {'mlabel2subclusters': {1: 2, 2: 1, 3: 3, 4: 1}},
 'seed_intensities': {'mus': tensor([109.6722, 220.9658, 100.9801,  38.6364, 125.5148, 108.1950, 216.1060,
          190.5462,  55.3930,  59.2667,  72.0628,  68.8775,  76.5113,  84.6639,
           90.0124,  94.1701,  67.0610,  25.9465,  31.5438,  21.0375, 192.4223,
          173.7434, 139.9284, 121.3904, 145.4289, 158.1318, 157.4630, 150.0894,
          183.9047, 181.7129, 114.8939,   9.5253,  29.0257,  97.9543, 122.0798,
           72.2969,  26.3086,  81.8050,  67.7463,  72.3737, 129.8539, 113.3900,
          141.8177, 225.0000,  35.3458, 173.7635,  29.5101, 135.9482, 188.2391,
          225.0000], device='cuda:0'),
  'sigmas': tensor([ 9.2432, 23.1060, 16.4965,  6.4289, 24.7862, 23.7996, 15.2424, 20.2845,
          12.6833,  6.9079,  6.1214, 22.1317,  9.7907,  5.5302, 14.3288, 11.1454,
          16.0453, 20.9057, 24.2358, 13.4785, 22.7258, 11.2053, 12.9420, 13.4270,
          14.8660, 22.4874,  5.6251,  9.8794,  8.8749, 19.0294,  9.7164,  6.2293,
          13.6376, 11.7447, 14.1414,  6.4362, 20.4575, 14.6729,  8.4719, 14.2926,
           6.9458, 11.5346, 14.6113,  6.6516, 22.1767,  8.3793, 20.1699,  6.3299,
           5.3340, 21.8027], device='cuda:0')},
 'deform_params': {'affine': {'rotations': array([ 0.0008224 ,  0.03067143, -0.0151502 ]),
   'shears': array([-0.01735838,  0.00744726,  0.00012507]),
   'scalings': array([1.09345725, 0.91695532, 0.98194215])},
  'non_rigid': {'nonlin_scale': array([0.05686841]),
   'nonlin_std': 1.048839010036788,
   'size_F_small': [15, 15, 15]},
  'flip': False},
 'gamma_params': {'gamma': 0.960299468352801},
 'bf_params': {'bf_scale': None, 'bf_std': None, 'bf_size': None},
 'resample_params': {'spacing': array([0.65685245, 0.65685245, 0.65685245])},
 'noise_params': {'noise_std': None},
 'generation_time': 0.5615839958190918}
```
</details>

<br>
If the `key:value` pair exists in the passed `genparams` dictionary, the `sample` method will use directly the value from the `genparams` dictionary. If the `key:value` pair does not exist in the `genparams` dictionary or it is `None`,  `sample` method will generate the value randomly, using the corresponding class attributes.

See how the keys `bf_scale`, `bf_std`, `bf_size` and `noise_std` have not been defined in the `genparams` dictionary above. This means that the `sample` method will generate these values randomly. The same could have been achieved by not passing them at all.

<details>
    {'idx': 0,
    'img_paths': PosixPath('../data/sub-sta38/anat/sub-sta38_rec-irtk_T2w.nii.gz'),
    'segm_paths': PosixPath('../data/sub-sta38/anat/sub-sta38_rec-irtk_T2w.nii.gz'),
    'seeds': defaultdict(dict,
                {1: {1: PosixPath('../data/derivatives/seeds/subclasses_1/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_1.nii.gz'),
                2: PosixPath('../data/derivatives/seeds/subclasses_1/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_2.nii.gz'),
                3: PosixPath('../data/derivatives/seeds/subclasses_1/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_3.nii.gz'),
                4: PosixPath('../data/derivatives/seeds/subclasses_1/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_4.nii.gz')},
                2: {1: PosixPath('../data/derivatives/seeds/subclasses_2/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_1.nii.gz'),
                2: PosixPath('../data/derivatives/seeds/subclasses_2/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_2.nii.gz'),
                3: PosixPath('../data/derivatives/seeds/subclasses_2/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_3.nii.gz'),
                4: PosixPath('../data/derivatives/seeds/subclasses_2/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_4.nii.gz')},
                3: {1: PosixPath('../data/derivatives/seeds/subclasses_3/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_1.nii.gz'),
                2: PosixPath('../data/derivatives/seeds/subclasses_3/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_2.nii.gz'),
                3: PosixPath('../data/derivatives/seeds/subclasses_3/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_3.nii.gz'),
                4: PosixPath('../data/derivatives/seeds/subclasses_3/sub-sta38/anat/sub-sta38_rec-irtk_T2w_dseg_mlabel_4.nii.gz')}}),
    'selected_seeds': {'mlabel2subclusters': {1: 2, 2: 1, 3: 3, 4: 1}},
    'seed_intensities': {'mus': tensor([109.6722, 220.9658, 100.9801,  38.6364, 125.5148, 108.1950, 216.1060,
            190.5462,  55.3930,  59.2667,  72.0628,  68.8775,  76.5113,  84.6639,
            90.0124,  94.1701,  67.0610,  25.9465,  31.5438,  21.0375, 192.4223,
            173.7434, 139.9284, 121.3904, 145.4289, 158.1318, 157.4630, 150.0894,
            183.9047, 181.7129, 114.8939,   9.5253,  29.0257,  97.9543, 122.0798,
            72.2969,  26.3086,  81.8050,  67.7463,  72.3737, 129.8539, 113.3900,
            141.8177, 225.0000,  35.3458, 173.7635,  29.5101, 135.9482, 188.2391,
            225.0000], device='cuda:0'),
    'sigmas': tensor([ 9.2432, 23.1060, 16.4965,  6.4289, 24.7862, 23.7996, 15.2424, 20.2845,
            12.6833,  6.9079,  6.1214, 22.1317,  9.7907,  5.5302, 14.3288, 11.1454,
            16.0453, 20.9057, 24.2358, 13.4785, 22.7258, 11.2053, 12.9420, 13.4270,
            14.8660, 22.4874,  5.6251,  9.8794,  8.8749, 19.0294,  9.7164,  6.2293,
            13.6376, 11.7447, 14.1414,  6.4362, 20.4575, 14.6729,  8.4719, 14.2926,
            6.9458, 11.5346, 14.6113,  6.6516, 22.1767,  8.3793, 20.1699,  6.3299,
            5.3340, 21.8027], device='cuda:0')},
    'deform_params': {'affine': {'rotations': array([ 0.0008224 ,  0.03067143, -0.0151502 ]),
    'shears': array([-0.01735838,  0.00744726,  0.00012507]),
    'scalings': array([1.09345725, 0.91695532, 0.98194215])},
    'non_rigid': {'nonlin_scale': array([0.05686841]),
    'nonlin_std': 1.048839010036788,
    'size_F_small': [15, 15, 15]},
    'flip': False},
    'gamma_params': {'gamma': 0.960299468352801},
    'bf_params': {'bf_scale': array([0.00797334]),
    'bf_std': array([0.21896995]),
    'bf_size': [2, 2, 2]},
    'resample_params': {'spacing': array([0.65685245, 0.65685245, 0.65685245])},
    'noise_params': {'noise_std': None},
    'generation_time': 0.6192283630371094}
    ```

</details>