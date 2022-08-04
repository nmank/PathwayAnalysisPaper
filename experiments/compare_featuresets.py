import mlflow
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

import seaborn as sns

def jaccard(list1 : list, list2: list) -> float:
    '''
    find the jaccard overlap
    '''
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def load_best_run(experiment: str, method_runs: pd.DataFrame) -> pd.DataFrame:
    '''
    index highest BSR mlflow run by an experiment
    '''
    query = method_runs['params.experiment'] == experiment
    exp_method_runs = method_runs[query]
    max_idx = exp_method_runs['metrics.train_test.Test.Mean.BSR'].idxmax()    
    best_run = exp_method_runs.loc[max_idx]

    return best_run

def get_max_config(df: pd.DataFrame, method: str) -> pd.DataFrame:
    '''
    get the best runs for a method by experiment
    '''
    query = df['params.method'] == method
    method_runs = df[query]

    best_runs = pd.DataFrame(columns = df.columns)

    for experiment in np.unique(method_runs['params.experiment']):

        best_run = load_best_run(experiment, method_runs)

        best_runs = best_runs.append(best_run, ignore_index = True)

    return best_runs

def get_precomputed_directed_pagerank(df: pd.DataFrame) -> pd.DataFrame:
    query = (df['params.method'] == 'CPE') &\
            (df['params.centrality_measure'] == 'page_rank') &\
            (df['params.similarity_measure'] == 'precomputed') &\
            (df['params.directed'] == 'True')
    good_cpe_runs = df[query]

    return good_cpe_runs

def get_pathways_glpe(exp_id: str, best_runs: pd.DataFrame) -> list:
    '''
    get the glpe pathways that were selected in feature selection
    '''
    # import IPython; IPython.embed()

    query = best_runs['params.experiment'] == exp_id
    run = best_runs[query]
    pathways = pd.read_csv(os.path.join(run.filter(regex='artifact').iloc[0,0], 'pathway_ranks.csv'), dtype=object).dropna(how = 'all')
    pathways['batch:Ranks'] = pathways['batch:Ranks'].astype(int)
    selected_pathways = pathways[pathways['batch:Selected'] == '1']
    pathway_list = list(selected_pathways['Unnamed: 0'])
    return pathway_list

def limma_query_switch(exp_id: str, cepa_runs: pd.DataFrame, similarity_measure: str, centrality_measure: str) -> pd.Series:
    '''
    generate a query for cepa using cpe centrality_measure, similarity_measure and exp_id
    ''' 
    query = (cepa_runs['params.centrality_measure'] == centrality_measure) & \
            (cepa_runs['params.similarity_measure'] == similarity_measure) & \
            (cepa_runs['params.experiment'] == exp_id) & \
            (cepa_runs['params.directed'] == 'False')

    return query

def get_cepa_query(best_cpe_runs: pd.DataFrame, cepa_runs: pd.DataFrame, exp_id: str) -> pd.Series:
    '''
    query cepa using cpe run and experiment id
    '''

    query = best_cpe_runs['params.experiment'] == exp_id
    cpe_run  = best_cpe_runs[query]
    similarity_measure = cpe_run['params.similarity_measure'].item()
    centrality_measure = cpe_run['params.centrality_measure'].item()

    query = limma_query_switch(exp_id, cepa_runs, similarity_measure, centrality_measure)

    return query

def get_pathways_cepa(best_cpe_runs: pd.DataFrame, exp_id: str, cepa_runs: pd.DataFrame, pval_thresh =.05) -> list:
    '''
    get cepa pathways using experiment, score, and p value threshhold
    '''
    query = get_cepa_query(best_cpe_runs, cepa_runs, exp_id)

    run = cepa_runs[query]

    pathways = pd.read_csv(os.path.join(run['artifact_uri'].item(), 'cpe_pathway_ranks.csv'), dtype=object)
    pathways['p_val'] = pathways['p_val'].astype(float)
    # pathways = pathways.sort_values(by = 'p_val')
    pathways = pathways.sort_values(by = 'score', ascending = False)
    best_pathways = pathways.query("p_val< @pval_thresh")
    cepa_pathways = list(best_pathways.iloc[:60]['ReactomeID'])

    return cepa_pathways

def generate_jaccard_matrix(cpe_pathway_list: list, lpe_pathway_list: list, cepa_pathways: list, ora_pathways: list, flu_pathways: list) -> pd.DataFrame:
    '''
    calculate the jaccard score between pathway lists
    '''
    all_pathways = [cpe_pathway_list]+[lpe_pathway_list]+[cepa_pathways]+[ora_pathways]

    label_names = [f'CPE ({len(cpe_pathway_list)})', 
                    f'LPE ({len(lpe_pathway_list)})', 
                    f'CePa ({len(cepa_pathways)})', 
                    f'ORA ({len(ora_pathways)})']

    if len(flu_pathways) > 0:
        all_pathways = all_pathways+[flu_pathways]

        label_names = label_names + [f'Flu ({len(flu_pathways)})']

    j_mat = np.zeros((len(all_pathways),len(all_pathways)))
    for i in range(len(all_pathways)):
        for j in range(len(all_pathways)):
            j_mat[i,j] = jaccard(all_pathways[i], all_pathways[j])

    
    data_to_plot = pd.DataFrame(data = j_mat, 
                                index = label_names,
                                columns = label_names)

    return data_to_plot

def intersection_across_cepa_cpe_lpe(exp_id: str, cepa_runs: pd.DataFrame, best_lpe_runs: pd.DataFrame, best_cpe_runs: pd.DataFrame) -> None:
    cpe_pathway_list = set(get_pathways_glpe(exp_id, best_cpe_runs))

    lpe_pathway_list = set(get_pathways_glpe(exp_id, best_lpe_runs))

    cepa_pathways = set(get_pathways_cepa(best_cpe_runs, exp_id, pval_thresh =.05))

    pathways_intersect = cpe_pathway_list.intersection(lpe_pathway_list.intersection(cepa_pathways))
    return list(pathways_intersect) 

def make_plot(exp_id: str, cepa_runs: pd.DataFrame, best_lpe_runs: pd.DataFrame, best_cpe_runs: pd.DataFrame, flu_pathways: list) -> None:
    '''
    plot a heatmap of the jaccard score by experiment id
    '''

    cpe_pathway_list = get_pathways_glpe(exp_id, best_cpe_runs)

    lpe_pathway_list = get_pathways_glpe(exp_id, best_lpe_runs)

    cepa_pathways = get_pathways_cepa(best_cpe_runs, exp_id, cepa_runs, pval_thresh =.05)
    try:
        ora_pathways_df = pd.read_csv(f'../ge_ora/{exp_id}.csv', index_col = 0)

        ora_query = ora_pathways_df['pvalue'] < 1 #1 pvalue threshold for ORA

        ora_pathways = set(list(ora_pathways_df[ora_query]['ID']))

        data_to_plot = generate_jaccard_matrix(cpe_pathway_list, lpe_pathway_list, cepa_pathways, ora_pathways, flu_pathways)

        plt.figure(exp_id)
        sns.heatmap(data_to_plot, annot=True, cmap = "rocket_r")
        plt.xticks(rotation=45) 
        plt.subplots_adjust(bottom=0.15)

        plt.savefig(f'../jaccard_plots/{exp_id}.png')
    except:
        print('no ora results')

def best_pathways(df: pd.DataFrame) -> None:
    best_pw_dict = {}
    for exp_id in df['params.experiment']:
        query = (df['params.experiment'] == exp_id) 
        run = df[query]
        pathways = pd.read_csv(os.path.join(run.filter(regex='artifact').iloc[0,0], 'pathway_ranks.csv'), dtype=object).dropna(how = 'all')
        best_idx = pathways['batch:absWeights'].astype('float').idxmax()
        best_pathway = pathways.iloc[best_idx][['Unnamed: 0','Name', 'n_probes','n_genes']]
        best_pw_dict[exp_id] = best_pathway
    best_pw = pd.DataFrame.from_dict(best_pw_dict, orient = 'index')
    best_pw.to_csv('../jaccard_plots/new_best_pathways.csv')

if __name__ == '__main__':

    glpe_runs = mlflow.search_runs(experiment_ids=['2']).fillna(value=np.nan)

    cepa_runs = mlflow.search_runs(experiment_ids=['0']).fillna(value=np.nan)


    flu_pathways = list(pd.read_csv('../preprocessing/reactome_flu_pathways.csv')['ReactomeID'])

    best_cpe_runs = get_precomputed_directed_pagerank(glpe_runs)
    best_lpe_runs = get_max_config(glpe_runs, 'LPE')


    for exp in best_lpe_runs['params.experiment']:
        make_plot(exp, cepa_runs, best_lpe_runs, best_cpe_runs, flu_pathways = flu_pathways)
        best_pathways(best_cpe_runs)
