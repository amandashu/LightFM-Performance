from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from scipy.sparse import coo_matrix

def run_lightfm(data):
    train_data = coo_matrix(data['train'])
    test_data = coo_matrix(data['test'])

    model = LightFM(loss='warp')
    model.fit(train_data, epochs=30, num_threads=2)
    print("Train precision: %.2f" % precision_at_k(model, train_data, k=5).mean())
    print("Test precision: %.2f" % precision_at_k(model, test_data, k=5).mean())
