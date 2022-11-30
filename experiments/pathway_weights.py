import mlflow
import pandas as pd
import seaborn as sns
import compare_featuresets as cf
import numpy as np
import matplotlib.pyplot as plt
import os

def get_pathways(exp_id, best_runs):
    query = best_runs['params.experiment'] == exp_id
    run = best_runs[query]
    pathways = pd.read_csv(os.path.join(run.filter(regex='artifact').iloc[0,0], 'pathway_ranks.csv'), dtype=object, index_col = 0).dropna(how = 'all')

    return pathways

if __name__ == '__main__':

    glpe_runs = mlflow.search_runs(experiment_ids=['2']).fillna(value=np.nan)

    best_cpe_runs = cf.get_precomputed_directed_pagerank(glpe_runs)
    best_lpe_runs = cf.get_max_config(glpe_runs, 'LPE')

    all_cpe_pathways = set()
    all_lpe_pathways = set()

    experiments = []
    for exp in best_lpe_runs['params.experiment']:
        if 'limma' in exp:
            cpe_pathways = cf.get_pathways_glpe(exp, best_cpe_runs)
            lpe_pathways = cf.get_pathways_glpe(exp, best_lpe_runs)

            all_cpe_pathways = all_cpe_pathways.union(set(cpe_pathways))
            all_lpe_pathways = all_lpe_pathways.union(set(lpe_pathways))

            experiments.append('_'.join(exp.split('_')[1:-2]))

    all_cpe_pathways = list(all_cpe_pathways)
    all_lpe_pathways = list(all_lpe_pathways)

    cpe_heat_map = pd.DataFrame(columns = all_cpe_pathways, index = experiments)
    lpe_heat_map = pd.DataFrame(columns = all_lpe_pathways, index = experiments)


    for exp in best_lpe_runs['params.experiment']:
        if 'limma' in exp:

            exp_index = '_'.join(exp.split('_')[1:-2])
            cpe_pathways = get_pathways(exp, best_cpe_runs)
            lpe_pathways = get_pathways(exp, best_lpe_runs)



            cpe_heat_map.loc[exp_index, all_cpe_pathways] = cpe_pathways.loc[ all_cpe_pathways, 'batch:absWeights']
            lpe_heat_map.loc[exp_index, all_lpe_pathways] = lpe_pathways.loc[ all_lpe_pathways, 'batch:absWeights']

    sorted_exp = ['4to2_1_8', '4to2_9_16', '4to2_17_24', '4to2_25_32', '6to1_1_8', '6to1_9_16', '6to1_17_24', '6to1_25_32']

    cpe_heat_map = cpe_heat_map.loc[sorted_exp]
    lpe_heat_map = lpe_heat_map.loc[sorted_exp]

    exp_map = {'4to2_1_8': '4 studies, 1 to 8hr', '4to2_9_16': '4 studies, 9 to 16hr', '4to2_17_24': '4 studies, 17 to 24hr', '4to2_25_32':'4 studies, 25 to 32hr', 
            '6to1_1_8': '6 studies, 1 to 8hr', '6to1_9_16': '6 studies, 9 to 16hr' , '6to1_17_24': '6 studies, 17 to 24hr', '6to1_25_32':'6 studies, 25 to 35hr'}

    cpe_heat_map = cpe_heat_map.rename(index=exp_map)
    lpe_heat_map = lpe_heat_map.rename(index=exp_map)
        
    cpe_heat_map = cpe_heat_map.astype(float)
    lpe_heat_map = lpe_heat_map.astype(float)

    ########################################################
    #indices of the best pathways

    cpe_ranks = cpe_heat_map.sum(axis = 0).sort_values()
    idx = cpe_ranks > .7
    best_cpe_pathways = list(cpe_ranks[idx].index)
    cpe_pathways.loc[best_cpe_pathways]['Name'].to_csv('./results/top_cpe_all_exp.csv', header = False)

    lpe_ranks = lpe_heat_map.sum(axis = 0).sort_values()
    idx = lpe_ranks > .7
    best_lpe_pathways = list(lpe_ranks[idx].index)
    lpe_pathways.loc[best_lpe_pathways]['Name'].to_csv('./results/top_lpe_all_exp.csv', header = False)

    ########################################################
    #weight plots

    fig,ax = plt.subplots()
    plt.plot(cpe_ranks)
    ax.set_xticklabels([])
    ax.set_xticks([])
    plt.ylabel('Total SVM Weights')
    plt.xlabel('Pathway')
    plt.title('CPE: SVM Weights')
    plt.savefig('./results/cpe_svm_weights.png')

    fig,ax = plt.subplots()
    plt.plot(lpe_ranks)
    ax.set_xticklabels([])
    ax.set_xticks([])
    plt.ylabel('Total SVM Weights')
    plt.xlabel('Pathway')
    plt.title('LPE: SVM Weights')
    plt.savefig('./results/lpe_svm_weights.png')

    ########################################################
    #heat map plots


    cpe_heat_map = cpe_heat_map.reindex(sorted(cpe_heat_map.columns), axis=1)
    all_pathways = list(cpe_heat_map.columns)
    cpe_show_ticks = []
    ii = 0
    for p in all_pathways:
        if p in best_cpe_pathways:
            cpe_show_ticks.append(ii)
        ii+=1
    plt.figure(figsize=(12,5))
    sns.set(font_scale=1.4)
    s = sns.heatmap(data = cpe_heat_map, cmap = "rocket_r", xticklabels = True)
    s.hlines([4], *s.get_xlim(), color = 'k')
    s.set_title(f'CPE SSVM Weights')
    s.set_xlabel('Pathways')
    fig = s.get_figure()
    s.tick_params(left=False, bottom = True)
    # plt.xticks(rotation=45)
    plt.xticks(np.array(cpe_show_ticks) + 0.5)
    plt.tight_layout()
    fig.savefig('./results/cpe_weights.png')

    lpe_heat_map = lpe_heat_map.reindex(sorted(lpe_heat_map.columns), axis=1)
    all_pathways = list(lpe_heat_map.columns)
    lpe_show_ticks = []
    ii = 0
    for p in all_pathways:
        if p in best_lpe_pathways:
            lpe_show_ticks.append(ii)
        ii+=1
    plt.figure(figsize=(12,5))
    sns.set(font_scale=1.4)
    s = sns.heatmap(data = lpe_heat_map, cmap = "rocket_r", xticklabels = True)
    s.hlines([4], *s.get_xlim(), color = 'k')
    s.set_title(f'LPE SSVM Weights')
    s.set_xlabel('Pathways')
    fig = s.get_figure()
    s.tick_params(left=False, bottom = True)
    # plt.xticks(rotation=45)
    plt.xticks(np.array(lpe_show_ticks) + 0.5)
    plt.tight_layout()
    fig.savefig('./results/lpe_weights.png')