from src.models.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.models.Base.NonPersonalizedRecommender import TopPop
from scipy.sparse import csr_matrix

def run_toppop(data):
    # get train, test data in sparse matrices
    train_data = csr_matrix(data['train'])
    test_data = csr_matrix(data['test'])

    # get test results
    evaluator_test = EvaluatorHoldout(test_data, cutoff_list=[5, 10, 20], exclude_seen=False)

    recommender = TopPop(train_data)
    recommender.fit()
    
    results_dict, results_run_string = evaluator_test.evaluateRecommender(recommender)
    print("Result of TopPop is:\n" + results_run_string)
