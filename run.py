
import json
import sys

sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/analysis')

from etl import get_data
from analysis import run_analysis

from analysis_topPopular import run_toppop
from analysis_itemknn_cf import run_itemknn_cf
from analysis_userknn_cf import run_userknn_cf
from analysis_p3alpha import run_p3alpha
from analysis_rp3beta import run_rp3beta
from analysis_LightFM import run_lightfm

import numpy as np
import pandas as pd

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis'.
    `main` runs the targets in order of data=>analysis.
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
    
    ### metrics ###
    with open('config/analysis-params.json') as fh:
        metrics_config = json.load(fh)
    metrics_to_get = metrics_config['metrics_to_optimize']
    
    ### all analysis targets ###
    if 'all' in targets:
        algo_names = ['toppop', 'itemknncf', 'userknncf', 'p3alpha', 'rp3beta', 'lightfm']
        config_dct = {}

        for a in algo_names:
            with open('config/' + a + '-params.json') as fh:
                analysis_cfg = json.load(fh)
            config_dct[a] = analysis_cfg 
            
        for metric in metrics_to_get:
            for a in algo_names[:len(algo_names)-1]:
                config_dct[a]["metric_to_optimize"] = metric
            run_analysis(data, config_dct)
        
        #TopPop
        dfs = []
        for metric in metrics_to_get:
            df = pd.read_csv('calculatedMetrics\Popular_' + metric + '.csv')
            dfs.append(df)
        combined_df = pd.concat(dfs, axis=1)
        combined_df.index = np.array(['TopPop'])
        print(combined_df)
        combined_df.to_csv('calculatedMetrics\Popular_metrics.csv', index=False)
        combined_df.to_latex('latexOutputs\Popular_results.tex')
        #ItemKNN
        dfs = []
        for metric in metrics_to_get:
            df = pd.read_csv('calculatedMetrics\itemknncf_' + metric + '.csv')
            dfs.append(df)
        combined_df = pd.concat(dfs, axis=1)
        combined_df.index = np.array(['ItemKNNCF'])
        print(combined_df)
        combined_df.to_csv('calculatedMetrics\itemknncf_metrics.csv', index=False)
        combined_df.to_latex('latexOutputs\itemknncf_results.tex')
        #UserKNN
        dfs = []
        for metric in metrics_to_get:
            df = pd.read_csv('calculatedMetrics\knnusercf_' + metric + '.csv')
            dfs.append(df)
        combined_df = pd.concat(dfs, axis=1)
        combined_df.index = np.array(['UserKNNCF'])
        print(combined_df)
        combined_df.to_csv('calculatedMetrics\knnusercf_metrics.csv', index=False)
        combined_df.to_latex('latexOutputs\knnusercf_results.tex')
        #P3alpha
        dfs = []
        for metric in metrics_to_get:
            df = pd.read_csv('calculatedMetrics\p3alpha_' + metric + '.csv')
            dfs.append(df)
        combined_df = pd.concat(dfs, axis=1)
        combined_df.index = np.array(['P3alpha'])
        print(combined_df)
        combined_df.to_csv('calculatedMetrics\p3alpha_metrics.csv', index=False)
        combined_df.to_latex('latexOutputs\p3alpha_results.tex')
        #RP3beta
        dfs = []
        for metric in metrics_to_get:
            df = pd.read_csv('calculatedMetrics\RP3beta_' + metric + '.csv')
            dfs.append(df)
        combined_df = pd.concat(dfs, axis=1)
        combined_df.index = np.array(['RP3beta'])
        print(combined_df)
        combined_df.to_csv('calculatedMetrics\RP3beta_metrics.csv', index=False)
        combined_df.to_latex('latexOutputs\RP3beta_results.tex')
        
        df_popular = pd.read_csv('calculatedMetrics\Popular_metrics.csv')
        df_itemknn = pd.read_csv('calculatedMetrics\itemknncf_metrics.csv')
        df_userknn = pd.read_csv('calculatedMetrics\knnusercf_metrics.csv')
        df_p3alpha = pd.read_csv('calculatedMetrics\p3alpha_metrics.csv')
        df_rp3beta = pd.read_csv('calculatedMetrics\RP3beta_metrics.csv')
        dfs = [df_popular, df_itemknn, df_userknn, df_p3alpha, df_rp3beta]
        dfs_index = np.array(['TopPop', 'ItemKNNCF', 'UserKNNCF', 'P3alpha', 'RP3beta'])
        baselines_df = pd.concat(dfs)
        baselines_df.index = dfs_index
        print(baselines_df)
        baselines_df.to_csv('calculatedMetrics\All_metrics.csv')
        baselines_df.to_latex('latexOutputs\All_results.tex')
        

    ### individual analysis target ###
    if 'toppop' in targets:
        with open('config/toppop-params.json') as fh:
            analysis_cfg = json.load(fh)
        
        for metric in metrics_to_get:
            analysis_cfg["metric_to_optimize"] = metric
            run_toppop(data, **analysis_cfg)
            
        dfs = []
        for metric in metrics_to_get:
            df = pd.read_csv('calculatedMetrics\Popular_' + metric + '.csv')
            dfs.append(df)
            
        combined_df = pd.concat(dfs, axis=1)
        combined_df.index = np.array(['TopPop'])
        print(combined_df)
        combined_df.to_csv('calculatedMetrics\Popular_metrics.csv', index=False)
        combined_df.to_latex('latexOutputs\Popular_results.tex')

    if 'itemknncf' in targets:
        with open('config/itemknncf-params.json') as fh:
            analysis_cfg = json.load(fh)
            
        for metric in metrics_to_get:
            analysis_cfg["metric_to_optimize"] = metric
            run_itemknn_cf(data, **analysis_cfg)
            
        dfs = []
        for metric in metrics_to_get:
            df = pd.read_csv('calculatedMetrics\itemknncf_' + metric + '.csv')
            dfs.append(df)
            
        combined_df = pd.concat(dfs, axis=1)
        combined_df.index = np.array(['ItemKNNCF'])
        print(combined_df)
        combined_df.to_csv('calculatedMetrics\itemknncf_metrics.csv', index=False)
        combined_df.to_latex('latexOutputs\itemknncf_results.tex')

    if 'userknncf' in targets:
        with open('config/userknncf-params.json') as fh:
            analysis_cfg = json.load(fh)
        
        for metric in metrics_to_get:
            analysis_cfg["metric_to_optimize"] = metric
            run_userknn_cf(data, **analysis_cfg)
            
        dfs = []
        for metric in metrics_to_get:
            df = pd.read_csv('calculatedMetrics\knnusercf_' + metric + '.csv')
            dfs.append(df)
            
        combined_df = pd.concat(dfs, axis=1)
        combined_df.index = np.array(['UserKNNCF'])
        print(combined_df)
        combined_df.to_csv('calculatedMetrics\knnusercf_metrics.csv', index=False)
        combined_df.to_latex('latexOutputs\knnusercf_results.tex')

    if 'p3alpha' in targets:
        with open('config/p3alpha-params.json') as fh:
            analysis_cfg = json.load(fh)
        
        for metric in metrics_to_get:
            analysis_cfg["metric_to_optimize"] = metric
            run_p3alpha(data, **analysis_cfg)
            
        dfs = []
        for metric in metrics_to_get:
            df = pd.read_csv('calculatedMetrics\p3alpha_' + metric + '.csv')
            dfs.append(df)
            
        combined_df = pd.concat(dfs, axis=1)
        combined_df.index = np.array(['P3alpha'])
        print(combined_df)
        combined_df.to_csv('calculatedMetrics\p3alpha_metrics.csv', index=False)
        combined_df.to_latex('latexOutputs\p3alpha_results.tex')


    if 'rp3beta' in targets:
        with open('config/rp3beta-params.json') as fh:
            analysis_cfg = json.load(fh)
        
        for metric in metrics_to_get:
            analysis_cfg["metric_to_optimize"] = metric
            run_rp3beta(data, **analysis_cfg)
            
        dfs = []
        for metric in metrics_to_get:
            df = pd.read_csv('calculatedMetrics\RP3beta_' + metric + '.csv')
            dfs.append(df)
            
        combined_df = pd.concat(dfs, axis=1)
        combined_df.index = np.array(['RP3beta'])
        print(combined_df)
        combined_df.to_csv('calculatedMetrics\RP3beta_metrics.csv', index=False)
        combined_df.to_latex('latexOutputs\RP3beta_results.tex')

    if 'lightfm' in targets:
        with open('config/lightfm-params.json') as fh:
            analysis_cfg = json.load(fh)

        run_lightfm(data, **analysis_cfg)

    return

if __name__ == '__main__':
    # run via:
    # python main.py data model
    targets = sys.argv[1:]
    main(targets)