
import json
import sys

sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/analysis')

from etl import get_data
from analysis import run_analysis

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis'.

    `main` runs the targets in order of data=>analysis.
    '''

    if 'data-local' in targets:
        with open('config/data-local-params.json') as fh:
            data_cfg = json.load(fh)

        # make the data target
        data = get_data(**data_cfg)

    if 'data-dsmlp' in targets:
        with open('config/data-dsmlp-params.json') as fh:
            data_cfg = json.load(fh)

        # make the data target
        data = get_data(**data_cfg)

    if 'analysis' in targets:
        with open('config/analysis-params.json') as fh:
            analysis_cfg = json.load(fh)

        # make the data target
        run_analysis(data, **analysis_cfg)

    return

if __name__ == '__main__':
    # run via:
    # python main.py data model
    targets = sys.argv[1:]
    main(targets)
