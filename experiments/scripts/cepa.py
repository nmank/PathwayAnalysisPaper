"""This script performs centrality ranking of pathways based on a set of selected features."""

#use this:  mlflow run -e feature-set-pathway-ranking --no-conda ./ --experiment-id 0
  

# imports
import argparse
from pandas import DataFrame
from pandas import Series
from pandas import Index
import glob
from numpy import ndarray
import pandas as pd
import numpy as np
import mlflow
import sys
sys.path.append('/home/katrina/a/mankovic/PathwayAnalysis')
from GLPE import CLPE
import os
import itertools

sys.path.append('/data4/mankovic/GSE73072/experiments/modules')
from utils import load_pathway_metadata, reformat_pathway_metadata


# set command line params
parser = argparse.ArgumentParser(description="This script performs centrality " \
                                             "expression of pathways based on a set of selected features.")
parser.add_argument("--data-dir", dest="data_dir", type=str,
                    help="The directory containing the normalized data files.")
parser.add_argument("--features-dir", dest="features_dir", type=str,
                    help="")
parser.add_argument("--pathway-file", dest="pathway_file", type=str,
                    help="")
parser.add_argument("--pathway-metadata", dest="pathway_metadata", type=str,
                    help="")
args = parser.parse_args()

def extract_pathway_genes(pw_matrices: str) -> pd.DataFrame:
    """
    Takes in a pathway-gene dataframe of shape (n_pathways, n_genes) and
    returns a dataframe with the gene ids for each pathways along with the number
    of genes in a given pathway.
    """

    pathway_names = []
    n_pids = []
    for f in os.listdir(pw_matrices):
        start = f.find("R-HSA")
        end = f.find(".csv")

        pathway_names.append(f[start:end])

        pw_matrix = pd.read_csv(pw_matrices +'/'+ f, index_col = 0)
        pids = list(pw_matrix.columns)
         
        n_pids.append(len(pids))



    gene_columns = ['n_pids'] + ['n_genes'] + [f"gid_{i}" for i in range(np.max(n_pids))]
    pathway_genes = pd.DataFrame(index = pathway_names, columns=gene_columns)

    ii = 0
    for f in os.listdir('/data3/darpa/omics_databases/ensembl2pathway/pathways_edges/pathways/pw_edge_mtx'):
        pw_matrix = pd.read_csv('/data3/darpa/omics_databases/ensembl2pathway/pathways_edges/pathways/pw_edge_mtx/' + f, index_col = 0)

        genes = [g[7:] for g in list(pw_matrix.columns)]

        n_genes = len(genes)

        pathway_genes.iloc[ii, 2:n_genes+2] = genes #maybe change this
        pathway_genes.iat[ii, 0] = n_pids[ii]
        pathway_genes.iat[ii, 1] = n_genes 
        ii+=1

    return pathway_genes

def group_data_paths_by_experiment(data_file_paths: list, feature_file_paths : list) -> list:
    """Groups data paths for train and test data paths for each experiment. """

    # strip train/test from ends
    feature_file_names = [os.path.basename(file_path).rstrip('.csv') for file_path in feature_file_paths]

    # initialize output
    out = []

    for experiment in feature_file_names:
        experiment_data_id = '_'.join(experiment.split('_')[2:])
        for data_file in data_file_paths:
            if (experiment_data_id in data_file) and ('train' in data_file):
                experiment_data = data_file
        for feature_file in feature_file_paths:
            if experiment in feature_file:
                experiment_features = feature_file
                

        # append to output
        out.append((experiment, experiment_data, experiment_features))

    return out

def append_pathway_metadata(ranked_pathways: DataFrame, pw_matrices: str, 
                            n_genes_total: int, pathway_metadata: DataFrame) -> DataFrame:

    pw_metadata_by_rid = pathway_metadata.groupby('ReactomeID').agg(pd.unique)

    pathway_genes = extract_pathway_genes(pw_matrices)

    pathway_info = pw_metadata_by_rid[['Name', 'url']]
    vardata = pd.concat((pathway_info, pathway_genes), axis=1).loc[pathway_genes.index]
    vardata.index.name = 'ReactomeID'

    ranked_pathways = ranked_pathways.rename(columns={'pathway':'ReactomeID'})
    ranked_pathways = ranked_pathways.set_index('ReactomeID')

    ranked_pathways = pd.concat((ranked_pathways, vardata), axis=1).loc[ranked_pathways.index]

    return ranked_pathways

def cepa_ranking(data: DataFrame, pathway_metadata: DataFrame,
                 pathway_file: str,
                 features: DataFrame, similarity_measure: str,
                 centrality_measure: str, directed: bool) -> None:
    # # logging

    # # generate the incidence matrix
    # incidence_matrix = generate_incidence_matrix()

    # initialize CLPE
    clpe = CLPE(centrality_measure = centrality_measure, 
                network_type = similarity_measure,
                feature_ids = list(data.columns),
                pathway_files = pathway_file,
                directed = directed,
                heat_kernel_param=2,
                normalize_rows=False)

    # fit CLPE
    clpe.fit(np.array(data))

    
    restricted_feat_ids = list(set(list(data.columns)).intersection(list(features['ProbeID'])))
    ranked_pathways = clpe.simple_transform(np.array(restricted_feat_ids), n_null_trials = 500)

    ranked_pathways = append_pathway_metadata(ranked_pathways,
                                            pathway_file, 
                                            len(list(data.columns)), 
                                            pathway_metadata)


    return ranked_pathways

##TO DO:
#determine the best pathways
#run svm test

# main function
def main() -> None:

    # get file paths
    data_file_paths = glob.glob(f"{args.data_dir}/*.csv")
    features_file_paths = glob.glob(f"{args.features_dir}/*.csv")

    pathway_file = args.pathway_file


    # generate train/test paths by experiment
    experiment_paths = group_data_paths_by_experiment(data_file_paths, features_file_paths)

    # load the pathway metadata
    pathway_metadata = load_pathway_metadata(args.pathway_metadata)
    pathway_metadata = reformat_pathway_metadata(pathway_file, pathway_metadata)

    similarity_measures = ['precomputed', 'correlation']
    centrality_measures = ['degree', 'page_rank']

    for experiment, data_path, features_path in experiment_paths:

        print(f'doing experiment: experiment')
        # loop through different methods
        param_combinations = itertools.product(similarity_measures,
                                                centrality_measures,
                                                [True, False],
                                                )

        for similarity_measure, centrality_measure, directed in param_combinations:

            # start and mlflow run
            with mlflow.start_run(experiment_id=os.environ['MLFLOW_EXPERIMENT_ID']):
                # log the parameters
                mlflow.log_param('experiment', experiment)
                mlflow.log_param('data_path', data_path)
                mlflow.log_param('features_path', features_path)
                mlflow.log_param('similarity_measure', similarity_measure)
                mlflow.log_param('centrality_measure', centrality_measure)
                mlflow.log_param('directed', str(directed))

                # set the data
                data = pd.read_csv(data_path, index_col=0)
                

                # set the features
                features = pd.read_csv(features_path, index_col=0)

                # run cepa
                ranked_pathways = cepa_ranking(data, pathway_metadata, pathway_file, features, similarity_measure, centrality_measure, directed = directed)

                # save artifact
                temp_path = '/data4/mankovic/GSE73072/experiments/temp/cpe_pathway_ranks.csv'
                ranked_pathways.to_csv(temp_path)
                mlflow.log_artifact(temp_path)

if __name__=="__main__":

    # run main
    main()

# def generate_incidence_matrix(pathway_file: str, dataset: DataFrame, similarity_measure: str):

#     if similarity_measure == 'precomputed':
#         pathway_edges = pd.read_csv(pathway_file, index_col = 0).dropna()

#         #convert source and destination feature ids to strings
#         pathway_edges['src'] = pathway_edges['src'].apply(str)
#         pathway_edges['dst'] = pathway_edges['dst'].apply(str)

#         #define a mapping from featureids to columns in the dataset
#         node_ids = list(dataset.columns)
#         translate_dict = { node_ids[i] :i  for i in range(len(node_ids))}

#         #restrict pathway edges to the features in the dataset
#         boolean_src = pathway_edges.src.isin(node_ids)
#         boolean_dst = pathway_edges.dst.isin(node_ids)
#         pathway_edges = pathway_edges[boolean_src & boolean_dst]

#         #generate a new pathway edges dataframe with only the features in the dataset
#         #and with location in the dataset rather than feature id
#         new_pathway_edges = pd.DataFrame(columns = ['src','dst', 'type', 'ReactomeID'])

#         new_pathway_edges['src'] = pathway_edges["src"].map(translate_dict)
#         new_pathway_edges['dst'] = pathway_edges["dst"].map(translate_dict)
#         new_pathway_edges['type'] = pathway_edges['type'] 
#         new_pathway_edges['ReactomeID'] =pathway_edges['ReactomeID']
#     else:
#         #THIS STILL NEEDS TO BE TESTED!

#         #which genes are in which pathways
#         pathway_edges = pd.read_csv(pathway_file)

#         #restrict pathway data to to genes that are actually there
#         pathway_edges = pathway_edges[['ReactomeID']+list(dataset.columns)]

#         new_pathway_edges=pd.DataFrame(columns = ['feature_id', 'ReactomeID'])
#         gene_names = pathway_edges.columns
#         for row in np.array(pathway_edges):
#             idx = np.where(row == True)
#             # print(row)
#             for g in gene_names[idx]:
#                 new_pathway_edges = new_pathway_edges.append({'feature_id': int(g), 'ReactomeID':row[0]}, ignore_index = True)


    # return np.array(new_pathway_edges)