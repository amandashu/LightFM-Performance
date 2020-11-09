from analysis_topPopular import run_toppop
from analysis_itemknn_cf import run_itemknn_cf
from analysis_userknn_cf import run_userknn_cf
from analysis_p3alpha import run_p3alpha
from analysis_rp3beta import run_rp3beta
#from analysis_LightFM import run_lightfm

def run_analysis(data, **kwargs):

    # run top popular
    print('\nRun top popular')
    #run_toppop(data)

    #run itemknn cf
    print('\nRun itemknn cf')
    #run_itemknn_cf(data)

    #run userknn cf
    print('\nRun userknn cf')
    run_userknn_cf(data)

    # run p3alpha
    print('\nRun p3alpha')
    #run_p3alpha(data)

    # run rp3beta
    print('\nRun rp3beta')
    run_rp3beta(data)

    # run lightfm
    print('\nRun lightfm')
    #run_lightfm(data)

    return None
