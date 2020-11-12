from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import recall_at_k
from scipy.sparse import coo_matrix
from sklearn.model_selection import ParameterGrid

def print_lightfm_metrics(data, model, cutoffs):
    print('\n--PRECISION--')
    for k in cutoffs:
        print("k = " + str(k))
        print("Train precision: %.2f" % precision_at_k(model, data['train'], k=k).mean())
        print("Test precision: %.2f" % precision_at_k(model, data['test'], k=k).mean())

    print('\n--RECALL--')
    for k in cutoffs:
        print("k = " + str(k))
        print("Train recall: %.2f" % recall_at_k(model, data['train'], k=k).mean())
        print("Test recall: %.2f" % recall_at_k(model, data['test'], k=k).mean())

def run_lightfm(data, **kwargs):
    # get data in sparse matrices
    for key in data:
        data[key] = coo_matrix(data[key])

    ##############
    ### tuning ###
    ##############

    # find best parameters
    best_precision = -1
    best_params = {}
    for g in ParameterGrid({key:val for key, val in kwargs.items() if key != 'cutoffs'}):
        model = LightFM(loss='warp',**g)
        model.fit(data['train_small'], epochs=30)
        precision = precision_at_k(model,data['validation']).mean()

        if precision > best_precision:
            best_precision = precision
            best_params = g
    print('best: ' + str(best_params))

    # fit with best parameters
    model = LightFM(loss='warp',**best_params)
    model.fit(data['train'], epochs=30)

    # print metrics
    print('\n###### LightFM - Tuned ######')
    print_lightfm_metrics(data, model,kwargs['cutoffs'])

    ###############
    ### untuned ###
    ###############

    # fit model with default parameters
    model = LightFM(loss='warp')
    model.fit(data['train'], epochs=30)

    # print metrics
    print('\n###### LightFM - Not Tuned ######')
    print_lightfm_metrics(data, model,kwargs['cutoffs'])
