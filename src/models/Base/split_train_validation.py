from src.models.Base.IncrementalSparseMatrix import IncrementalSparseMatrix
import numpy as np
def split_train_validation_percentage_user_wise(URM_train, train_percentage = 0.1, verbose=True):


    # ensure to use csr matrix or we get big problem
    URM_train = URM_train.tocsr()


    num_users, num_items = URM_train.shape

    URM_train_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items)
    URM_validation_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items)

    user_no_item_train = 0
    user_no_item_validation = 0

    for user_id in range(URM_train.shape[0]):

        start_pos = URM_train.indptr[user_id]
        end_pos = URM_train.indptr[user_id+1]


        user_profile_items = URM_train.indices[start_pos:end_pos]
        user_profile_ratings = URM_train.data[start_pos:end_pos]
        user_profile_length = len(user_profile_items)

        n_train_items = round(user_profile_length*train_percentage)

        if n_train_items == len(user_profile_items) and n_train_items > 1:
            n_train_items -= 1

        indices_for_sampling = np.arange(0, user_profile_length, dtype=np.int)
        np.random.shuffle(indices_for_sampling)

        train_items = user_profile_items[indices_for_sampling[0:n_train_items]]
        train_ratings = user_profile_ratings[indices_for_sampling[0:n_train_items]]

        validation_items = user_profile_items[indices_for_sampling[n_train_items:]]
        validation_ratings = user_profile_ratings[indices_for_sampling[n_train_items:]]

        if len(train_items) == 0:
            if verbose: print("User {} has 0 train items".format(user_id))
            user_no_item_train += 1

        if len(validation_items) == 0:
            if verbose: print("User {} has 0 validation items".format(user_id))
            user_no_item_validation += 1


        URM_train_builder.add_data_lists([user_id]*len(train_items), train_items, train_ratings)
        URM_validation_builder.add_data_lists([user_id]*len(validation_items), validation_items, validation_ratings)

    if user_no_item_train != 0:
        print("Warning split: {} users with 0 train items ({} total users)".format(user_no_item_train, URM_train.shape[0]))
    if user_no_item_validation != 0:
        print("Warning split: {} users with 0 validation items ({} total users)".format(user_no_item_validation, URM_train.shape[0]))

    URM_train = URM_train_builder.get_SparseMatrix()
    URM_validation = URM_validation_builder.get_SparseMatrix()


    return URM_train, URM_validation