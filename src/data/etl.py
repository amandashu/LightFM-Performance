from lightfm.datasets import fetch_movielens

def get_data(**kwargs):
    data = fetch_movielens(**kwargs)
    return data
