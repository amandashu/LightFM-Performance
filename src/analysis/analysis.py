from analysis_topPopular import run_toppop
from analysis_itemknn_cf import run_itemknn_cf
from analysis_userknn_cf import run_userknn_cf
from analysis_p3alpha import run_p3alpha
from analysis_rp3beta import run_rp3beta
from analysis_LightFM import run_lightfm
from analysis_LightFM_hybrid import run_lightfm_hybrid

def run_analysis(data, num_iterations, config_dct):

    # run top popular
    print('\nRun top popular')
    run_toppop(data, **config_dct['toppop'])

    #run itemknn cf
    print('\nRun itemknn cf')
    run_itemknn_cf(data, num_iterations, **config_dct['itemknncf'])

    #run userknn cf
    print('\nRun userknn cf')
    run_userknn_cf(data, num_iterations, **config_dct['userknncf'])

    # run p3alpha
    print('\nRun p3alpha')
    run_p3alpha(data, num_iterations, **config_dct['p3alpha'])

    # run rp3beta
    print('\nRun rp3beta')
    run_rp3beta(data, num_iterations, **config_dct['rp3beta'])

    # run lightfm pure
    print('\nRun lightfm')
    run_lightfm(data, **config_dct['lightfm'])

    # run lightfm hybrid
    print('\nRun lightfm hybrid')
    run_lightfm_hybrid(data, **config_dct['lightfm-hybrid'])
    return None
