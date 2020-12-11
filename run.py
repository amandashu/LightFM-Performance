
import json
import sys

sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/analysis')
sys.path.insert(0, 'src/utils')

from etl import get_data
from analysis import run_analysis
from analysis_topPopular import run_toppop
from analysis_itemknn_cf import run_itemknn_cf
from analysis_userknn_cf import run_userknn_cf
from analysis_p3alpha import run_p3alpha
from analysis_rp3beta import run_rp3beta
from analysis_LightFM import run_lightfm
from analysis_LightFM_hybrid import run_lightfm_hybrid
from report import report
from clean import remove_results

import numpy as np
import pandas as pd

def all(data, num_iterations=35):
    """
    Runs all the algorithms, given data
    """
    algo_names = ['toppop', 'itemknncf', 'userknncf', 'p3alpha', 'rp3beta', 'lightfm', 'lightfm-hybrid']
    config_dct = {}

    for a in algo_names:
        with open('config/' + a + '-params.json') as fh:
            analysis_cfg = json.load(fh)
        config_dct[a] = analysis_cfg
    run_analysis(data, num_iterations, config_dct)

    # create report
    with open('config/report-params.json') as fh:
        report_cfg = json.load(fh)
    report(**report_cfg)

def data_target(target):
    """
    Runs the data pipeline, given the data target
    """
    with open('config/' + target + '-params.json') as fh:
        data_cfg = json.load(fh)

    data = get_data(**data_cfg)
    return data

def algo_target(data, target, num_iterations=35):
    """
    Runs the model algorithm, given the algorithm target
    """
    with open('config/' + target + '-params.json') as fh:
        analysis_cfg = json.load(fh)

    if target == 'toppop':
        run_toppop(data, **analysis_cfg)
    elif target == 'itemknncf':
        run_itemknn_cf(data, num_iterations, **analysis_cfg)
    elif target == 'userknncf':
        run_userknn_cf(data, num_iterations, **analysis_cfg)
    elif target == 'p3alpha':
        run_p3alpha(data, num_iterations, **analysis_cfg)
    elif target == 'rp3beta':
        run_rp3beta(data, num_iterations, **analysis_cfg)
    elif target == 'lightfm':
        run_lightfm(data, **analysis_cfg)
    elif target == 'lightfm-hybrid':
        run_lightfm_hybrid(data, **analysis_cfg)

    # create report
    with open('config/report-params.json') as fh:
        report_cfg = json.load(fh)
    report(**report_cfg)

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    '''
    ### standard targets
    if 'clean' in targets:
        remove_results()

    if 'all' in targets: # defaults to dsmlp data
        data = data_target('data-dsmlp')
        all(data)
        return

    if 'test' in targets:
        data = data_target('data-test')
        num_iterations = 5
        all(data, num_iterations)
        return

    ### data targets ###
    if 'data-local' in targets:
        data = data_target('data-local')

    if 'data-dsmlp' in targets:
        data = data_target('data-dsmlp')

    ### all analysis targets ###
    if 'all-algos' in targets:
        all(data)

    ### individual analysis target ###
    if 'toppop' in targets:
        algo_target(data, 'toppop')

    if 'itemknncf' in targets:
        algo_target(data, 'itemknncf')

    if 'userknncf' in targets:
        algo_target(data, 'userknncf')

    if 'p3alpha' in targets:
        algo_target(data, 'p3alpha')

    if 'rp3beta' in targets:
        algo_target(data, 'rp3beta')

    if 'lightfm' in targets:
        algo_target(data, 'lightfm')

    if 'lightfm-hybrid' in targets:
        algo_target(data, 'lightfm-hybrid')
    return

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
