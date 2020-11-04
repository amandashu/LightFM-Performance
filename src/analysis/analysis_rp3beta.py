from src.models.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.models.GraphBased.RP3betaRecommender import RP3betaRecommender
from scipy.sparse import csr_matrix


def run_rp3beta(data):
    train_data = csr_matrix(data['train'])
    test_data = csr_matrix(data['test'])

    evaluator_test = EvaluatorHoldout(test_data, cutoff_list=[5, 10, 20], exclude_seen=False)

    recommender = RP3betaRecommender(train_data)
    recommender.fit(alpha=1., beta=0.6) # needs to be tuned

    results_dict, results_run_string = evaluator_test.evaluateRecommender(recommender)
    print("Result of rp3beta is:\n" + results_run_string)
