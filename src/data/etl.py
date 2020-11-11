import pandas as pd
import numpy as np

num_items = 1682
num_users = 943

def user_item_interactions(path, min_rating):
    df = pd.read_csv(path,'\t', names=['user id','item id','rating','timestamp'])
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
    dct = {}
    dct['train'] = user_item_interactions(kwargs['train_path'], kwargs['min_rating'])
    dct['test'] = user_item_interactions(kwargs['test_path'], kwargs['min_rating'])
    return dct
