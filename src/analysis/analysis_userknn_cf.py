from functools import partial
import numpy as np
from scipy.sparse import csr_matrix
from src.models.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.models.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from src.models.ParameterTuning.run_parameter_search import runParameterSearch_Collaborative
import re
import ast
import pandas as pd
import os

def run_userknn_cf(data, metrics_to_optimize, cutoffs):
    # get data in sparse matrices
    data['train'] = csr_matrix(data['train'])
    data['test'] = csr_matrix(data['test'])
    data['train_small'] = csr_matrix(data['train_small'])
    data['validation'] = csr_matrix(data['validation'])

    # get results of tuned baseline
    evaluator_validation = EvaluatorHoldout(data['validation'], cutoff_list=cutoffs, exclude_seen=False)
    evaluator_test = EvaluatorHoldout(data['test'], cutoff_list=cutoffs, exclude_seen=False)
    dfs_for_metrics = []

    for metric in metrics_to_optimize:
        metric_to_optimize = metric
        runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train = data['train_small'],
                                                       URM_train_last_test = None,
                                                       metric_to_optimize = metric_to_optimize,
                                                       evaluator_validation_earlystopping = evaluator_validation,
                                                       evaluator_validation = evaluator_validation,
                                                       evaluator_test = evaluator_test,
                                                       parallelizeKNN = False,
                                                       allow_weighting = True,
                                                       resume_from_saved = True,
                                                       n_cases = 35,
                                                       n_random_starts = 5)

        try:
            runParameterSearch_Collaborative_partial(UserKNNCFRecommender)
        except Exception as e:
            print("On recommender {} Exception {}".format(UserKNNCFRecommender, str(e)))
            traceback.print_exc()

        similarities = ['asymmetric', 'cosine', 'dice', 'jaccard', 'tversky']
        sim_config_dict = {}

        # Get best configuration with each similarity
        for sim in similarities:
            config = ""
            with open("result_experiments/UserKNNCFRecommender_" + metric_to_optimize + "_" + sim + '_SearchBayesianSkopt.txt') as f:
                for line in f:
                    pass
                config = ast.literal_eval(re.search('({.+})', line).group(0))
                sim_config_dict[sim] = config

        #Find metrics for each similarity and cutoff
        sim_metric_dict = {}
        for sim in similarities:
            tuning = sim_config_dict[sim]
            recommender = UserKNNCFRecommender(data['train'])

            if sim == 'cosine' or sim == 'asymmetric':
                recommender.fit(topK = tuning['topK'], shrink = tuning['shrink'], similarity=tuning['similarity'], normalize=tuning['normalize'], feature_weighting=tuning['feature_weighting'])

            elif sim == 'tversky':
                recommender.fit(topK = tuning['topK'], shrink = tuning['shrink'], similarity=tuning['similarity'], normalize=tuning['normalize'], tversky_alpha=tuning['tversky_alpha'], tversky_beta = tuning['tversky_beta'])

            else:
                recommender.fit(topK = tuning['topK'], shrink = tuning['shrink'], similarity=tuning['similarity'], normalize=tuning['normalize'])

            results_dict, results_run_string = evaluator_test.evaluateRecommender(recommender)

            metric = {}
            for cutoff in cutoffs:
                metric[cutoff] = results_dict[cutoff][metric_to_optimize]
            sim_metric_dict[sim] = metric


        # Find best metric for each cutoff
        cutoff_metrics = {}
        cutoff_configs = {}
        for cutoff in cutoffs:
            max_metric = 0
            best_config = ""
            for sim in similarities:
                metric = sim_metric_dict[sim][cutoff]
                if metric > max_metric:
                    max_metric = metric
                    best_config = sim_config_dict[sim]
            cutoff_metrics[cutoff] = max_metric
            cutoff_configs[cutoff] = best_config

        metric_cols = []
        for cutoff in cutoff_metrics.keys():
            metric_cols.append(metric_to_optimize + '@' + str(cutoff))

        metric_table = pd.DataFrame(np.array([list(cutoff_metrics.values())]), columns=metric_cols)
        #print(metric_table)
        dfs_for_metrics.append(metric_table)

    combined_df = pd.concat(dfs_for_metrics, axis=1)
    combined_df.insert(0, 'Recommender', np.array(['UserKNNCF']))
    print(combined_df)

    # add results folder if it doesn't exist
    if not os.path.exists('results/'):
        os.makedirs('results/')

    try:
        all_df = pd.read_csv('results/Metrics.csv')
        dfs_index = list(all_df['Recommender'].values)
        combined_df_index = list(combined_df['Recommender'].values)[0]
        if combined_df_index in dfs_index:
            for col in all_df.columns:
                all_df.loc[all_df['Recommender'] == combined_df_index, col] = list(combined_df[col].values)[0]
        else:
            all_df = pd.concat([all_df, combined_df])
        all_df = all_df.reset_index(drop=True)
        print(all_df)
        all_df.to_csv('results/Metrics.csv', index=False)
        all_df.to_latex('results/Metrics.tex')
    except Exception as e:
        combined_df.to_csv('results/Metrics.csv', index=False)
        combined_df.to_latex('results/Metrics.tex')
