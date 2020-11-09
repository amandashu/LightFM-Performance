## DSC180-LightFM-Replication

This repository contains code for our DSC180 replication project of the LightFM model.

## Data
The data used is the [Movielens 100k dataset](https://grouplens.org/datasets/movielens/100k/).

See [here](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt) for a detailed description of the data. We use the files:
- ua.base: training data
- ua.test: testing data
- u.item: item info
- u.user: user info

## Code Organization

#### Configuration
The "config" folder contains two json files:
- analysis-params.json: contains parameters that are passed into the run_analysis function in analysis.py
- data-local-params.json: contains data parameters (when running locally) that are passed into the get_data function in etl.py
- data-dsmlp-params.json: contains data parameters (when on dsmlp) that are passed into the get_data function in etl.py

#### Source
The "src" folder contains subfolders "data", "analysis", and "models".

In the "src/data" folder:
- etl.py: contains the function "get_data" that reads in the MovieLens data and outputs training and testing user/item interaction matrixes

In the "src/analysis" folder:
- analysis.py: contains the function "run_analysis" to run the results of lightfm and baseline algorithms
- All other files starting with "analysis_" contain a function that runs their respective algorithm

The "src/models" folder contains baseline algorithms and evaluation code, taken from this [repository](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation) by Dacrema, an author of "Are we Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches".

## Run the Results
On DSMLP, run this command to get results:
```console
python run.py data-dsmlp analysis
```

On local, run this command to get results (this assumes there is a folder called "data", which contains required data files):
```console
python run.py data-local analysis
```

## Contributions
Checkpoint 1:
- Amanda: data ingestion pipeline
- Sarat: introduction
