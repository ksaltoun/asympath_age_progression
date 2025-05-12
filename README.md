Longitudinal changes in brain asymmetry track lifestyle and disease
==============================

This repository contains the code associated with the manuscript entitled 'Longitudinal changes in brain asymmetry track lifestyle and disease'. It  builds upon asymmetry patterns described in ["Dissociable brain structural asymmetry patterns reveal unique phenome-wide profiles"](https://doi.org/10.1038/s41562-022-01461-0). It explores the progression of brain asymmetry in middle-aged adults and connects brain asymmetry changes to a variety of lifestyle domains; behavioural and lifestyle changes, and clinically-diagnosed illnesses.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── processing         <- Functions which are used for visualization and data processing
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train linear regressions used throughout the work
    │   │   │                 Includes code to visualize and analyze regression coefs as wll
    │   │   └──linear_regression
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
