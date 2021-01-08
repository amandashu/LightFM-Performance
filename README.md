## DSC180-LightFM-Replication

This repository contains code for our DSC180 replication project of the LightFM model.

## Data
The data used is the [Movielens 100k dataset](https://grouplens.org/datasets/movielens/100k/).

See [here](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt) for a detailed description of the data. We use the files:
- `ua.base`: training data
- `ua.test`: testing data
- `u.item`: item features data

## Code Organization

#### Configuration
The `config` folder contains several json files:
- `data-local-params.json`: contains data parameters (when running locally) that are passed into the `get_data` function in `etl.py`
- `data-dsmlp-params.json`: contains data parameters (when on dsmlp) that are passed into the `get_data` function in `etl.py`
- `data-test-params.json`: contains data parameters for testing data when running the `test` target
- `report-params.json`: contains paramaters for building the `report.html`
- all other files in the format `<algorithm name>-params.json` contain the parameters passed into their respective functions

#### Source
The `src` folder contains subfolders `data`, `analysis`, `models`, and `utils`.

In the `src/data` folder:
- `etl.py`: contains the function `get_data` that reads in the MovieLens100k data and outputs training/validation/testing user-item interaction matrices and item features data

In the `src/analysis` folder:
- `analysis.py`: contains the function `run_analysis` to run the results of lightfm and baseline algorithms
- All other files starting with `analysis_` contain a function that runs their respective algorithm

The `src/models` folder contains baseline algorithms and evaluation code, taken from this [repository](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation) by Dacrema, an author of "Are we Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches".

In the `src/utils` folder:
- `report.py`: contains function `report` to save metric figures in `results/` and outputs `report.html` in `report/`. This file contains code modified from [here](https://github.com/DSC-Capstone/project-templates/blob/EDA/src/utils.py)
- `clean.py`: contains function `remove_results` that implements the standard target `clean`

#### Notebook
The `notebook` folder contains a Jupyter Notebook file that is run when building the report.

#### References
The `references` folder contains our citations.

#### Test
The `test/testdata` folder contains the testing data that is utilized for the standard target `test`. This data is only used to check correctness of the pipeline.

## Run the Results

On local, run this command to run all the algorithms and get the results (this assumes there is a folder called `data`, which contains required data files):
```console
python run.py data-local all-algos
```

Besides `all-algos`, you can specify which algorithms to run if you only want to get the results of certain ones. Possible targets include `toppop`, `itemknncf`, `userknncf`, `p3alpha`, `rp3beta`, `lightfm`, and `lightfm-hybrid`. The code below runs the p3alpha baseline algorithm:
```console
python run.py data-local p3alpha
```

Standard targets are also implemented. `all` will run all the algorithms and defaults to using data on DSMLP. `clean` will delete the folders that are outputted after running. `test` will run all the algorithms, using the testing data.

Running any baseline algorithm will create a folder `result_experiments`, which holds several files related to the tuning of the baseline algorithms.

Running any number of algorithms will create a folder `results` which contains:
- `Metrics.csv`: contains the metrics at each cutoff for each of the algorithms
- `Metrics.tex`: same as Metrics.csv but the table is in latex format
- `precision.png`/`recall.png`: line plots comparing algorithms by their metrics over each cutoff

Additionally, a folder `report` will contain `report.html`, which is the html version of `report.ipynb` that lies in the `notebook` folder.
