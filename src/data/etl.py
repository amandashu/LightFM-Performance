import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

num_items = 1682
num_users = 943

def read_datafiles(train_path, test_path):
    train_df = pd.read_csv(train_path,'\t', names=['user id','item id','rating','timestamp'])
    test_df = pd.read_csv(test_path,'\t', names=['user id','item id','rating','timestamp'])
    return train_df, test_df

def user_item_interactions(df, min_rating):
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

def get_data(**kwargs):
    train_df, test_df = read_datafiles(kwargs['train_path'], kwargs['test_path'])

    # split the training to its own training and validation set
    train_smalldf, validation_df = train_test_split(train_df, test_size=0.1)

    dct = {}
    mr = kwargs['min_rating']
    dct['train'] = user_item_interactions(train_df, mr)
    dct['test'] = user_item_interactions(test_df, mr)
    dct['train_small'] = user_item_interactions(train_smalldf, mr)
    dct['validation'] = user_item_interactions(validation_df, mr)
    return dct