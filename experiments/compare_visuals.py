import mlflow
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import os
import shutil

experiment_name = 'gse73072_4to2_9_16_subjectID_limma'

runs_PE = mlflow.search_runs(experiment_ids=['2']).fillna(value=np.nan)
runs_GE = mlflow.search_runs(experiment_ids=['3']).fillna(value=np.nan)

def get_max_config(df, method):
    query = df['params.method'] == method
    method_runs = df[query]

    best_runs = pd.DataFrame(columns = method_runs.columns)

    for experiment in np.unique(method_runs['params.experiment']):
        query = method_runs['params.experiment'] == experiment
        exp_method_runs = method_runs[query]
        max_idx = exp_method_runs['metrics.train_test.Test.Mean.BSR'].idxmax()    
        best_run = exp_method_runs.loc[max_idx]
        best_runs = best_runs.append(best_run, ignore_index = True)

    return best_runs


best_runs = pd.DataFrame(columns = runs_PE.columns)
best_runs = best_runs.append(get_max_config(runs_PE, 'CPE'), ignore_index = True)
print(len(best_runs))
best_runs = best_runs.append(get_max_config(runs_PE, 'LPE'), ignore_index = True)
print(len(best_runs))
best_runs = best_runs.append(runs_GE[runs_GE['params.method'] == 'GE'], ignore_index = True)
print(len(best_runs))


query = (best_runs['params.method'] == 'CPE') & \
        (best_runs['params.experiment'] == experiment_name)
best_run = best_runs[query]

rank_file_name = 'test_pe_selected.png' #or pathway_ranks

image_path = os.path.join(best_run['artifact_uri'].item(), rank_file_name)

# print(best_run['run_id'].item())
# img = mpimg.imread(image_path)
# plt.figure()
# imgplot = plt.imshow(img)
# # plt.title(best_run['params.method'].item())
# plt.axis('off')
# plt.savefig('/data4/mankovic/GSE73072/pca_plots/CPE_plot_test_selected.png')

shutil.copyfile(image_path[7:], '../pca_plots/CPE_plot_test_selected.png')

rank_file_name = 'test_pe.png' #or pathway_ranks

image_path = os.path.join(best_run['artifact_uri'].item(), rank_file_name)

# print(best_run['run_id'].item())
# img = mpimg.imread(image_path)
# plt.figure()
# imgplot = plt.imshow(img)
# # plt.title(best_run['params.method'].item())
# plt.axis('off')
# plt.savefig('/data4/mankovic/GSE73072/pca_plots/CPE_plot_test.png')

shutil.copyfile(image_path[7:], '../pca_plots/CPE_plot_test.png')


query = (best_runs['params.method'] == 'GE') & \
        (best_runs['params.experiment'] == experiment_name)
best_run = best_runs[query]


rank_file_name = 'test_ge_selected.png' #or pathway_ranks

image_path = os.path.join(best_run['artifact_uri'].item(), rank_file_name)

# print(best_run['run_id'].item())
# img = mpimg.imread(image_path)
# plt.figure()
# imgplot = plt.imshow(img)
# plt.axis('off')
# # plt.title(best_run['params.method'].item())
# plt.savefig('/data4/mankovic/GSE73072/pca_plots/GE_plot_test_selected.png')

shutil.copyfile(image_path[7:], '../pca_plots/GE_plot_test_selected.png')


rank_file_name = 'test_ge.png' #or pathway_ranks

image_path = os.path.join(best_run['artifact_uri'].item(), rank_file_name)

# print(best_run['run_id'].item())
# img = mpimg.imread(image_path)
# plt.figure()
# imgplot = plt.imshow(img)
# plt.axis('off')
# # plt.title(best_run['params.method'].item())
# plt.savefig('/data4/mankovic/GSE73072/pca_plots/GE_plot_test.png')

shutil.copyfile(image_path[7:], '../pca_plots/GE_plot_test.png')
