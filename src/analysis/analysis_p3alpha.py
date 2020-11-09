from functools import partial
import numpy as np
from scipy.sparse import csr_matrix
from src.models.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.models.GraphBased.P3alphaRecommender import P3alphaRecommender
from src.models.ParameterTuning.run_parameter_search import runParameterSearch_Collaborative

def run_p3alpha(data):
    # get train, test data in sparse matrices
    train_data = csr_matrix(data['train'])
    test_data = csr_matrix(data['test'])

    # create validation data
    index = np.arange(np.shape(data['train'])[0])
    np.random.shuffle(index)
    validation_data = csr_matrix(data['train'][index, :])

    # tune baselines, optimizing precision
    evaluator_validation = EvaluatorHoldout(validation_data, cutoff_list=[5, 10, 20], exclude_seen=False)
    evaluator_test = EvaluatorHoldout(test_data, cutoff_list=[5, 10, 20], exclude_seen=False)

    metric_to_optimize = 'PRECISION'
    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train = train_data,
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
        runParameterSearch_Collaborative_partial(P3alphaRecommender)
    except Exception as e:
        print("On recommender {} Exception {}".format(P3alphaRecommender, str(e)))
        traceback.print_exc()

    # get test results without tuning
    recommender = P3alphaRecommender(train_data)
    recommender.fit()
    results_dict, results_run_string = evaluator_test.evaluateRecommender(recommender)
    print("Result of p3alpha is:\n" + results_run_string)
