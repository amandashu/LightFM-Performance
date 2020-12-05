import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from lightfm.data import Dataset

genre_names =['unknown', 'Action','Adventure' , 'Animation' ,
          "Children's" , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
          'Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
          'Thriller' , 'War' , 'Western']

def read_datafiles(train_path, test_path, item_path):
    col_names = ['user id','item id','rating','timestamp']
    train_df = pd.read_csv(train_path,'\t', names=col_names)
    test_df = pd.read_csv(test_path,'\t', names=col_names)

    col_names = ['movie id' , 'movie title' , 'release date' , 'video release date' ,
              'IMDb URL'] + genre_names
    item_df = pd.read_csv( item_path,sep='|', names=col_names, usecols = ['movie id'] + genre_names, encoding='latin-1')
    return train_df, test_df, item_df

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

def get_item_features(train_df,test_df, item_df):
    # list of tuples (movie id, list of genres)
    item_featuers_iterable = list(zip(
        list(item_df['movie id']),
        list(item_df.iloc[:,1:].apply(lambda x:list(x[x==1].index), axis=1))
        ))

    full_data = pd.concat([train_df,test_df])
    dataset = Dataset()
    dataset.fit(users = full_data['user id'],
            items = full_data['item id'],
            item_features = genre_names
           )
    item_features = dataset.build_item_features(item_featuers_iterable)
    return item_features

def get_data(**kwargs):
    train_df, test_df, item_df = read_datafiles(kwargs['train_path'], kwargs['test_path'], kwargs['item_path'])

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
    dct['item_features'] = get_item_features(train_df, test_df, item_df)
    return dct
