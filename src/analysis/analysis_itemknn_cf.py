from src.models.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.models.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from scipy.sparse import csr_matrix

def run_itemknn_cf(data):
    train_data = csr_matrix(data['train'])
    test_data = csr_matrix(data['test'])

    evaluator_test = EvaluatorHoldout(test_data, cutoff_list=[5, 10, 20], exclude_seen=False)

    recommender = ItemKNNCFRecommender(train_data) # needs to be tuned
    recommender.fit()

    results_dict, results_run_string = evaluator_test.evaluateRecommender(recommender)
    print("Result of itemknn_cf is:\n" + results_run_string)
