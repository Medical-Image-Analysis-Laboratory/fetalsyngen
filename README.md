# Install the minimal conda environment for the project using fetalsyngen using conda (recommended)
```shell
conda env create -f environment.yml
```
Or using pip
```shell
pip install -r requirements.txt
```


# syngen
Synthetic data generator based on the domain randomization idea.

Run `pip install -e /home/vzalevskyi/projects/fetalsyngen` to set-up the package.

### docs

```shell
# install documentation tool
pip install 'mkdocstrings[python]'
# local deploy
mkdocs serve --watch ./fetalsyngen
# github pages deploy
mkdocs gh-deploy
```