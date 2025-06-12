# FetalSynGen
FetalSynthGen is a synthetic data generator created to address the challenges of limited data and domain shifts in fetal brain MRI analysis. It is based on the domain randomization approach of SynthSeg [1], which uses anatomical segmentations to create diverse synthetic images.


## Installation (local)
1. Install the minimal conda environment for the project using fetalsyngen using conda (recommended)
```shell
conda env create -f environment.yml
```
2. Activate the environment
```shell
conda activate fetalsyngen
```
3. Install the package
```shell
git clone https://github.com/Medical-Image-Analysis-Laboratory/fetalsyngen
cd fetalsyngen
pip install -e .
```
Please refer to [the documentation](https://medical-image-analysis-laboratory.github.io/fetalsyngen/) for more information.


### Documentation edit set-up

```shell
# install documentation tool
pip install 'mkdocstrings[python]'
# local deploy
mkdocs serve --watch ./fetalsyngen
# github pages deploy
mkdocs gh-deploy
```