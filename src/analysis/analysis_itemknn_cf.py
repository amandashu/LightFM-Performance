from functools import partial
import os, traceback, argparse
import numpy as np
from src.models.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.models.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from scipy.sparse import csr_matrix
from src.models.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from src.models.ParameterTuning.run_parameter_search import runParameterSearch_Collaborative

def run_itemknn_cf(data):
    train_data = csr_matrix(data['train'])
    test_data = csr_matrix(data['test'])
    print(data['train'])
    index = np.arange(np.shape(data['train'])[0])
    np.random.shuffle(index)
    validation_data = csr_matrix(data['train'][index, :])
    
    evaluator_validation = EvaluatorHoldout(validation_data, cutoff_list=[5, 10, 20], exclude_seen=False)
    evaluator_test = EvaluatorHoldout(test_data, cutoff_list=[5, 10, 20], exclude_seen=False)

    recommender = ItemKNNCFRecommender(train_data)# needs to be tuned
    recommender.fit()
    
    metric_to_optimize = 'NDCG' 
    
    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train = train_data,
                                                       URM_train_last_test = None,
                                                       metric_to_optimize = metric_to_optimize,
                                                       evaluator_validation_earlystopping = evaluator_validation,
                                                       evaluator_validation = evaluator_validation,
                                                       evaluator_test = evaluator_test,
                                                       output_folder_path = 'itemknnparameters',
                                                       parallelizeKNN = False,
                                                       allow_weighting = True,
                                                       resume_from_saved = True,
                                                       n_cases = 35,
                                                       n_random_starts = 5)
    
    try:
        runParameterSearch_Collaborative_partial('Item')
    except Exception as e:
        print("On recommender {} Exception {}".format(ItemKNNCFRecommender, str(e)))
        traceback.print_exc()
            
    results_dict, results_run_string = evaluator_test.evaluateRecommender(recommender)
    print("Result of itemknn_cf is:\n" + results_run_string)
