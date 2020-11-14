
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
from report import report

import numpy as np
import pandas as pd

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    '''
    ### data targets ###
    if 'data-local' in targets:
        with open('config/data-local-params.json') as fh:
            data_cfg = json.load(fh)

        data = get_data(**data_cfg)

    if 'data-dsmlp' in targets:
        with open('config/data-dsmlp-params.json') as fh:
            data_cfg = json.load(fh)

        data = get_data(**data_cfg)

    ### all analysis targets ###
    if 'all' in targets:
        algo_names = ['toppop', 'itemknncf', 'userknncf', 'p3alpha', 'rp3beta', 'lightfm']
        config_dct = {}

        for a in algo_names:
            with open('config/' + a + '-params.json') as fh:
                analysis_cfg = json.load(fh)
            config_dct[a] = analysis_cfg
        run_analysis(data, config_dct)

        # create report
        with open('config/report-params.json') as fh:
            report_cfg = json.load(fh)
        report(**report_cfg)

    ### individual analysis target ###
    if 'toppop' in targets:
        with open('config/toppop-params.json') as fh:
            analysis_cfg = json.load(fh)

        run_toppop(data, **analysis_cfg)

        # create report
        with open('config/report-params.json') as fh:
            report_cfg = json.load(fh)
        report(**report_cfg)

    if 'itemknncf' in targets:
        with open('config/itemknncf-params.json') as fh:
            analysis_cfg = json.load(fh)

        run_itemknn_cf(data, **analysis_cfg)

        # create report
        with open('config/report-params.json') as fh:
            report_cfg = json.load(fh)
        report(**report_cfg)

    if 'userknncf' in targets:
        with open('config/userknncf-params.json') as fh:
            analysis_cfg = json.load(fh)

        run_userknn_cf(data, **analysis_cfg)

        # create report
        with open('config/report-params.json') as fh:
            report_cfg = json.load(fh)
        report(**report_cfg)

    if 'p3alpha' in targets:
        with open('config/p3alpha-params.json') as fh:
            analysis_cfg = json.load(fh)

        run_p3alpha(data, **analysis_cfg)

        # create report
        with open('config/report-params.json') as fh:
            report_cfg = json.load(fh)
        report(**report_cfg)

    if 'rp3beta' in targets:
        with open('config/rp3beta-params.json') as fh:
            analysis_cfg = json.load(fh)

        run_rp3beta(data, **analysis_cfg)

        # createreport
        with open('config/report-params.json') as fh:
            report_cfg = json.load(fh)
        report(**report_cfg)

    if 'lightfm' in targets:
        with open('config/lightfm-params.json') as fh:
            analysis_cfg = json.load(fh)

        run_lightfm(data, **analysis_cfg)

        # create report
        with open('config/report-params.json') as fh:
            report_cfg = json.load(fh)
        report(**report_cfg)
    return

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
