from src.models.Base.Evaluation.Evaluator import EvaluatorHoldout
from scipy.sparse import csr_matrix
from src.models.GraphBased.P3alphaRecommender import P3alphaRecommender

def run_p3alpha(data):
    train_data = csr_matrix(data['train'])
    test_data = csr_matrix(data['test'])

    evaluator_test = EvaluatorHoldout(test_data, cutoff_list=[5, 10, 20], exclude_seen=False)

    recommender = P3alphaRecommender(train_data)
    recommender.fit(topK=100, alpha=0.5) # need to tune this

    results_dict, results_run_string = evaluator_test.evaluateRecommender(recommender)
    print("Result of p3alpha is:\n" + results_run_string)
