from src.models.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.models.Base.NonPersonalizedRecommender import TopPop
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np

def run_toppop(data, metric_to_optimize, cutoffs):
    # get train, test data in sparse matrices
    train_data = csr_matrix(data['train'])
    test_data = csr_matrix(data['test'])

    # get test results
    evaluator_test = EvaluatorHoldout(test_data, cutoff_list=cutoffs, exclude_seen=False)

    recommender = TopPop(train_data)
    recommender.fit()
    
    results_dict, results_run_string = evaluator_test.evaluateRecommender(recommender)
    
    cutoff_metrics = {}
    for key in results_dict.keys():
        cutoff_metrics[key] = results_dict[key][metric_to_optimize]
   
    metric_cols = []
    for cutoff in cutoff_metrics.keys():
        metric_cols.append(metric_to_optimize + '@' + str(cutoff))
            
    metric_table = pd.DataFrame(np.array([list(cutoff_metrics.values())]), columns=metric_cols)
    metric_table.index = np.array(['TopPop'])
    print(metric_table)
    metric_table.to_csv('calculatedMetrics\Popular_' + metric_to_optimize + '.csv', index=False)
