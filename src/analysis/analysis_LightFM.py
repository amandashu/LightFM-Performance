
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import recall_at_k
from scipy.sparse import coo_matrix
from sklearn.model_selection import ParameterGrid
import pandas as pd
import numpy as np

def run_lightfm(data, **kwargs):
    # get data in sparse matrices
    for key in data:
        data[key] = coo_matrix(data[key])

    ##############
    ### tuning ###
    ##############

    # find best parameters
    best_precision = -1
    best_params = {}
    for g in ParameterGrid({key:val for key, val in kwargs.items() if key != 'cutoffs' and key != 'metrics_to_optimize'}):
        model = LightFM(loss='warp',**g)
        model.fit(data['train_small'], epochs=30)
        precision = precision_at_k(model,data['validation']).mean()

        if precision > best_precision:
            best_precision = precision
            best_params = g
    print('best: ' + str(best_params))

    # fit with best parameters
    model_tuned = LightFM(loss='warp',**best_params)
    model_tuned.fit(data['train'], epochs=30)

    metrics_to_optimize = kwargs['metrics_to_optimize']
    cutoffs = kwargs['cutoffs']

    dfs_for_metrics = []
    for metric in metrics_to_optimize:
        metric_to_optimize = metric

        cutoff_metrics = {}
        if metric_to_optimize == "PRECISION":
            for cutoff in cutoffs:
                cutoff_metrics[cutoff] = precision_at_k(model_tuned, data['test'], k=cutoff).mean()

        elif metric_to_optimize == "RECALL":
            for cutoff in cutoffs:
                cutoff_metrics[cutoff] = recall_at_k(model_tuned, data['test'], k=cutoff).mean()

        metric_cols = []
        for cutoff in cutoff_metrics.keys():
            metric_cols.append(metric_to_optimize + '@' + str(cutoff))

        metric_table = pd.DataFrame(np.array([list(cutoff_metrics.values())]), columns=metric_cols)
        #print(metric_table)
        dfs_for_metrics.append(metric_table)

    combined_df = pd.concat(dfs_for_metrics, axis=1)
    combined_df.insert(0, 'Recommender', np.array(['LightFM']))
    print(combined_df)

    try:
        all_df = pd.read_csv('results\Metrics.csv')
        dfs_index = list(all_df['Recommender'].values)
        combined_df_index = list(combined_df['Recommender'].values)[0]
        if combined_df_index in dfs_index:
            for col in all_df.columns:
                all_df.loc[all_df['Recommender'] == combined_df_index, col] = list(combined_df[col].values)[0]
        else:
            all_df = pd.concat([all_df, combined_df])
        all_df = all_df.reset_index(drop=True)
        print(all_df)
        all_df.to_csv('results\Metrics.csv', index=False)
        all_df.to_latex('results\Metrics.tex')
    except Exception as e:
        combined_df.to_csv('results\Metrics.csv', index=False)
        combined_df.to_latex('results\Metrics.tex')
