import mlflow
import pandas as pd
import numpy as np
import seaborn as sns
import compare_featuresets as cf


glpe_runs = mlflow.search_runs(experiment_ids=['2']).fillna(value=np.nan)

cepa_runs = mlflow.search_runs(experiment_ids=['0']).fillna(value=np.nan)

best_cpe_runs = cf.get_precomputed_directed_pagerank(glpe_runs)
best_lpe_runs = cf.get_max_config(glpe_runs, 'LPE')

overlap_data = pd.DataFrame(columns = ['Method','Time Bin', 'Jaccard Overlap', 'Intersection', 'Modified Overlap'])
for t_bin in ['1_8','9_16', '17_24', '25_32']:
    exp_id4 = f'gse73072_4to2_{t_bin}_subjectID_limma'
    exp_id6 = f'gse73072_6to1_{t_bin}_subjectID_limma'

    cpe4_pathway_list = cf.get_pathways_glpe(exp_id4, best_cpe_runs)
    lpe4_pathway_list = cf.get_pathways_glpe(exp_id4, best_lpe_runs)

    cpe6_pathway_list = cf.get_pathways_glpe(exp_id6, best_cpe_runs)
    lpe6_pathway_list = cf.get_pathways_glpe(exp_id6, best_lpe_runs)

    ora_pathways_df = pd.read_csv(f'../ge_ora/{exp_id4}.csv', index_col = 0)
    ora_query = ora_pathways_df['pvalue'] < 1 #1 pvalue threshold for ORA
    ora4_pathway_list = set(list(ora_pathways_df[ora_query]['ID']))

    ora_pathways_df = pd.read_csv(f'../ge_ora/{exp_id6}.csv', index_col = 0)
    ora_query = ora_pathways_df['pvalue'] < 1 #1 pvalue threshold for ORA
    ora6_pathway_list = set(list(ora_pathways_df[ora_query]['ID']))

    cepa4_pathway_list = cf.get_pathways_cepa(best_cpe_runs, exp_id4, cepa_runs, pval_thresh =.05)
    cepa6_pathway_list = cf.get_pathways_cepa(best_cpe_runs, exp_id6, cepa_runs, pval_thresh =.05)


    cpe_overlap = cf.jaccard(cpe4_pathway_list,cpe6_pathway_list)
    lpe_overlap = cf.jaccard(lpe4_pathway_list,lpe6_pathway_list)
    cepa_overlap = cf.jaccard(cepa4_pathway_list, cepa6_pathway_list)
    ora_overlap = cf.jaccard(ora4_pathway_list,ora6_pathway_list)

    cpe_intersection = len(list(set(cpe4_pathway_list).union(cpe6_pathway_list)))
    lpe_intersection = len(list(set(lpe4_pathway_list).union(lpe6_pathway_list)))
    cepa_intersection = len(list(set(cepa4_pathway_list).union(cepa6_pathway_list)))
    ora_intersection = len(list(set(ora4_pathway_list).union(ora6_pathway_list)))

    cpe_overlap_better = cpe_overlap*(1-(cpe_intersection)/2261)
    lpe_overlap_better = lpe_overlap*(1-(lpe_intersection)/2261)
    cepa_overlap_better = cepa_overlap*(1-(cepa_intersection)/2261)
    ora_overlap_better = ora_overlap*(1-(ora_intersection)/2261)

    row = pd.DataFrame(columns = overlap_data.columns, data = [['CPE', t_bin, cpe_overlap, cpe_intersection, cpe_overlap_better]])
    overlap_data = pd.concat([overlap_data, row])
    row = pd.DataFrame(columns = overlap_data.columns, data = [['LPE', t_bin, lpe_overlap, lpe_intersection, lpe_overlap_better]])
    overlap_data = pd.concat([overlap_data, row])
    row = pd.DataFrame(columns = overlap_data.columns, data = [['CEPA', t_bin, cepa_overlap, cepa_intersection, cepa_overlap_better]])
    overlap_data = pd.concat([overlap_data, row])
    row = pd.DataFrame(columns = overlap_data.columns, data = [['ORA', t_bin, ora_overlap, ora_intersection, ora_overlap_better]])
    overlap_data = pd.concat([overlap_data, row])

overlap_data.to_csv('./results/overlap_across_experiments.csv')