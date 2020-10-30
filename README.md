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
- data-params.json: contains parameters that are passed into the get_data function in etl.py

#### Source
The "src" folder contains subfolders "data", "analysis", and "models".

In the "src/data" folder:
- etl.py: contains the function "get_data" that reads in the MovieLens data and outputs training and testing user/item interaction matrixes

In the "src/analysis" folder:
- analysis.py: contains the function "run_analysis" to run the results of lightfm and baseline algorithms on given inputted data

The "src/models" folder (in progress) contains baseline algorithms code.

## Run the Results
Run this command to get results:
```console
python run.py data analysis
```
## Contributions
Checkpoint 1:
- Amanda: data ingestion pipeline
- Sarat: introduction