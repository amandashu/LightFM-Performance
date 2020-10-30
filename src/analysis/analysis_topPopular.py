from src.data.Movielens100KReader import Movielens100KReader
from src.data.Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from src.models.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.models.Base.NonPersonalizedRecommender import TopPop

# Use a dataReader to load the data into sparse matrices
data_reader = Movielens100KReader()
loaded_dataset = data_reader.load_data()

# In the following way you can access the entire URM and the dictionary with all ICMs
URM_all = loaded_dataset.get_URM_all()
ICM_dict = loaded_dataset.get_loaded_ICM_dict()

# Create a training-validation-test split, for example by leave-1-out
# This splitter requires the DataReader object and the number of elements to holdout
dataSplitter = DataSplitter_leave_k_out(data_reader, k_out_value=1, use_validation_set=True)

# The load_data function will split the data and save it in the desired folder.
# Once the split is saved, further calls to the load_data will load the splitted data ensuring you always use the same split
dataSplitter.load_data(save_folder_path= "result_experiments/usage_example/data/")

# We can access the three URMs with this function and the ICMs (if present in the data Reader)
URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

ICM_dict = dataSplitter.get_loaded_ICM_dict()


# Now that we have the split, we can create the evaluators.
# The constructor of the evaluator allows you to specify the evaluation conditions (data, recommendation list length,
# excluding already seen items). Whenever you want to evaluate a model, use the evaluateRecommender function of the evaluator object
evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[5], exclude_seen=False)
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[5, 10, 20], exclude_seen=False)


# We now fit and evaluate a non personalized algorithm
recommender = TopPop(URM_train)
recommender.fit()

results_dict, results_run_string = evaluator_validation.evaluateRecommender(recommender)
print("Result of TopPop is:\n" + results_run_string)