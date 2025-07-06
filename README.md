# Installation
With this command, you can install the package directly from GitHub:
```
pip install pbdl
```

Update the package with this command:
```
pip install --upgrade pbdl
```

You can also install the *unstable* development version:
```
pip install git+https://github.com/tum-pbs/pbdl-dataset@develop
```

# Quick Guide
A quick guide with examples can be found in the [wiki](https://github.com/tum-pbs/pbdl-dataset/wiki/Quick-Guide). The code examples from the quick guide can also be found in [this](doc/pbdl-quick-guide-examples.ipynb) Jupyter notebook.

# Public datasets
All currently available public datasets are listed on [this](https://pfistse.github.io/dataset-gallery/) page.

# Testing
You can run the tests with the following command:
```
python -m unittest discover
```
Note that you need an internet connection as the tests require global datasets that you may not have cached.