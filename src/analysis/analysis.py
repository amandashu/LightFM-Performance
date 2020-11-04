from analysis_LightFM import run_lightfm
from analysis_topPopular import run_toppop

def run_analysis(data, **kwargs):

    # run top popular
    print('\nRun top popular')
    run_toppop(data)

    # run lightfm
    print('\nRun lightfm')
    run_lightfm(data)

    return None
