from functools import partial
import numpy as np
from scipy.sparse import csr_matrix
from src.models.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.models.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from src.models.ParameterTuning.run_parameter_search import runParameterSearch_Collaborative

def run_itemknn_cf(data, metric_to_optimize, cutoffs):
    # get data in sparse matrices
    for key in data:
        data[key] = csr_matrix(data[key])

    # get results of tuned baseline
    evaluator_validation = EvaluatorHoldout(data['validation'], cutoff_list=cutoffs, exclude_seen=False)
    evaluator_test = EvaluatorHoldout(data['test'], cutoff_list=cutoffs, exclude_seen=False)

    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train = data['train_small'],
                                                       URM_train_last_test = None,
                                                       metric_to_optimize = metric_to_optimize,
                                                       evaluator_validation_earlystopping = evaluator_validation,
                                                       evaluator_validation = evaluator_validation,
                                                       evaluator_test = evaluator_test,
                                                       parallelizeKNN = False,
                                                       allow_weighting = True,
                                                       resume_from_saved = True,
                                                       n_cases = 35,
                                                       n_random_starts = 5)

    try:
        runParameterSearch_Collaborative_partial(ItemKNNCFRecommender)
    except Exception as e:
        print("On recommender {} Exception {}".format(ItemKNNCFRecommender, str(e)))
        traceback.print_exc()

    # get test results without tuning
    recommender = ItemKNNCFRecommender(data['train'])
    recommender.fit()
    results_dict, results_run_string = evaluator_test.evaluateRecommender(recommender)
    print("Result of itemknn_cf is:\n" + results_run_string)
