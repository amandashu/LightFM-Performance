from src.models.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.models.Base.NonPersonalizedRecommender import TopPop
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np

def run_toppop(data, metrics_to_optimize, cutoffs):
    # get train, test data in sparse matrices
    train_data = csr_matrix(data['train'])
    test_data = csr_matrix(data['test'])

    # get test results
    evaluator_test = EvaluatorHoldout(test_data, cutoff_list=cutoffs, exclude_seen=False)

    recommender = TopPop(train_data)
    recommender.fit()
    
    results_dict, results_run_string = evaluator_test.evaluateRecommender(recommender)
    
    dfs_for_metrics = []
    for metric in metrics_to_optimize:
        metric_to_optimize = metric
        
        cutoff_metrics = {}
        for cutoff in cutoffs:
            cutoff_metrics[cutoff] = results_dict[cutoff][metric_to_optimize]
   
        metric_cols = []
        for cutoff in cutoff_metrics.keys():
            metric_cols.append(metric_to_optimize + '@' + str(cutoff))
            
        metric_table = pd.DataFrame(np.array([list(cutoff_metrics.values())]), columns=metric_cols)
        print(metric_table)
        dfs_for_metrics.append(metric_table)
    
    combined_df = pd.concat(dfs_for_metrics, axis=1)
    combined_df.insert(0, 'Recommender', np.array(['TopPop']))
    print(combined_df)
    
    try:
        all_df = pd.read_csv('calculatedMetrics\Metrics.csv')
        dfs_index = list(all_df['Recommender'].values)
        combined_df_index = list(combined_df['Recommender'].values)[0]
        if combined_df_index in dfs_index:
            for col in all_df.columns:
                all_df.loc[all_df['Recommender'] == combined_df_index, col] = list(combined_df[col].values)[0]
        else:
            all_df = pd.concat([all_df, combined_df])
        all_df = all_df.reset_index(drop=True)
        print(all_df)
        all_df.to_csv('calculatedMetrics\Metrics.csv', index=False)
        all_df.to_latex('latexOutputs\Metrics.tex')
    except Exception as e:
        combined_df.to_csv('calculatedMetrics\Metrics.csv', index=False)
        combined_df.to_latex('latexOutputs\Metrics.tex')