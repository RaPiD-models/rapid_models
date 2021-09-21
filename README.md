# rapid-models
Python package (Reciprocal Data and Physics models - RaPiD-models) to support more specific, accurate and timely decision support in operation of safety-critical systems, by combining physics-based modelling with data-driven machine learning and probabilistic uncertainty assessment.


* Free software: GNU General Public License v3
* Documentation: https://rapid-models.readthedocs.io.



## Quickstart
```sh
$ git clone https://github.com/RaPiD-models/rapid_models.git
$ cd rapid_models
$ pip install -e .
$ rapid_models --help
```


To develop, test, generate documentation, etc.
```sh
$ pip install -r requirements_dev.txt
```
    

To generate documentation do, either:
```sh
$ cd docs
$ make docs html 
```
or
```sh
$ cd docs
$ sphinx-build -M html . build
```
The html documentation will then be avaliable in `docs/build/html/index.html`


## Features
FIXME: add features


## Credits
This package was created with Cookiecutter and the `audreyr/cookiecutter-pypackage` project template.
* [Cookiecutter](https://github.com/audreyr/cookiecutter)
* [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage) 
