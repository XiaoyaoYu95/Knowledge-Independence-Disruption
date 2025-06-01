# Knowledge Independence Breeds Disruption but Limits Recognition

[![DOI](https://zenodo.org/badge/991603851.svg)](https://doi.org/10.5281/zenodo.15534384)

This repository contains data and code required to reproduce the findings in paper "Knowledge Independence Breeds Disruption but Limits Recognition". This repository is organized in the following way:

- A data file is anonymized if it contains identifiers that could reveal the identity of papers/authors; all analyzed data can be found in the `data\` directory.
- Notebooks and code related to the data preparing, analysis and visualization can be found in the `notebooks\` repository.
- `results\figures\` contains the visualization of the main results.
- `results\results_for_tables\` contains the analysis results for the tables in main context.


## System and software requirements

The project was created and executed on the the High Performance Computing Center from Southwest University and New York University Abu Dhabi. The code in this repo can run on any commercial laptop with Python 3 installed.

Python libraries:

- pandas 2.2.2
- numpy 1.26.4
- scipy 1.13.1
- seaborn 0.13.2
- matplotlib 3.9.2
- statsmodels 0.14.2
- psmpy 0.3.13
- patsy 0.5.6


Each Jupyter Notebook is self-contained and can be executed using a Notebook Server. The expected outputs are in the Jupyter Notebooks themselves, as well as contained in the `results\` directory.
