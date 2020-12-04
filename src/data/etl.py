import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

def read_datafiles(train_path, test_path, item_path, genre_path):
    col_names = ['user id','item id','rating','timestamp']
    train_df = pd.read_csv(train_path,'\t', names=col_names)
    test_df = pd.read_csv(test_path,'\t', names=col_names)

    col_names =['movie id' , 'movie title' , 'release date' , 'video release date' ,
              'IMDb URL' , 'unknown', 'Action','Adventure' , 'Animation' ,
              "Children's" , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
              'Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
              'Thriller' , 'War' , 'Western']
    item_df = pd.read_csv( item_path, sep='|', names=col_names, encoding='latin-1')
    col_names = ['Genre', 'GenreID']
    genre_df = pd.read_csv(genre_path, sep='|', names=col_names, encoding='latin-1')
    return train_df, test_df, item_df, genre_df

def user_item_interactions(df, min_rating, num_items, num_users):
    df = df[df['rating']>=min_rating]
    df_piv = df.pivot(index='user id',columns='item id',values='rating')
    cols = list(df_piv.columns)
    rows = list(df_piv.index)

    # fill in missing items
    missing_items = []
    for i in np.arange(1,num_items+1):
        if i not in cols:
            missing_items.append(i)

    # fill in missing users
    missing_users = []
    for i in np.arange(1,num_users+1):
        if i not in rows:
            missing_users.append(i)

    # empty columns for missing items
    for i in missing_items:
        df_piv[i] = np.NaN
    df_piv = df_piv.reindex(sorted(df_piv.index), axis=0) # sort

    # empty rows for missing users
    df_piv = df_piv.append(pd.DataFrame(index=missing_users,columns=list(np.arange(1,num_items+1))))
    df_piv = df_piv.sort_index() # sort
    df_piv = df_piv.fillna(0)

    return df_piv.to_numpy()

def parse_item_data(num_items, item_metadata_raw, genres_raw):
    genres = []
    
    for index, row in genres_raw.iterrows():
        genre = row['Genre']
        genres.append("genre:{}".format(genre))
    
    id_feature_labels = np.empty(num_items, dtype=np.object)
    genre_feature_labels = np.array(genres)

    id_features = sp.identity(num_items, format="csr", dtype=np.float32)
    genre_features = sp.lil_matrix((num_items, len(genres)), dtype=np.float32)
    
    for index, row in item_metadata_raw.iterrows():
            
        # Zero-based indexing
        iid = int(row['movie id']) - 1
        title = row['movie title']

        id_feature_labels[iid] = title
        
        genres_row = list(row.values)[5:]
        item_genres = [idx for idx, val in enumerate(genres_row) if int(val) > 0]

        for gid in item_genres:
            genre_features[iid, gid] = 1.0

    return (id_features, id_feature_labels, genre_features.tocsr(), genre_feature_labels)

def get_features(id_features, id_feature_labels, genre_features_matrix, genre_feature_labels, num_items):
    
    assert id_features.shape == (num_items, len(id_feature_labels))
    assert genre_features_matrix.shape == (num_items, len(genre_feature_labels))
    
    indicator_features = True
    genre_features = False
    if indicator_features and not genre_features:
        features = id_features
        feature_labels = id_feature_labels
    elif genre_features and not indicator_features:
        features = genre_features_matrix
        feature_labels = genre_feature_labels
    else:
        features = sp.hstack([id_features, genre_features_matrix]).tocsr()
        feature_labels = np.concatenate((id_feature_labels, genre_feature_labels))
    return (features, feature_labels)

def get_data(**kwargs):
    train_df, test_df, item_df, genre_df = read_datafiles(kwargs['train_path'], kwargs['test_path'], kwargs['item_path'], kwargs['genre_path'])
    # split the training to its own training and validation set
    train_smalldf, validation_df = train_test_split(train_df, test_size=0.1)

    dct = {}
    mr = kwargs['min_rating']
    ni = kwargs['num_items']
    nu = kwargs['num_users']
    dct['train'] = user_item_interactions(train_df, mr, ni, nu)
    dct['test'] = user_item_interactions(test_df, mr, ni, nu)
    dct['train_small'] = user_item_interactions(train_smalldf, mr, ni, nu)
    dct['validation'] = user_item_interactions(validation_df, mr, ni, nu)
    
    # Load metadata features
    (id_features, id_feature_labels, genre_features_matrix, genre_feature_labels) = parse_item_data(ni, item_df, genre_df)
    (features, feature_labels) = get_features(id_features, id_feature_labels, genre_features_matrix, genre_feature_labels, ni)
    

    dct['item_features'] = features
    dct['item_features_labels'] = feature_labels
    return dct
