"""General purpose module"""

# imports
import os
import numpy as np
import pandas as pd
from typing import Tuple, List
from pandas import DataFrame
from orthrus.core.dataset import DataSet

# functions
def generate_datasets(train_data_path: str, test_data_path: str, metadata: DataFrame) -> Tuple[DataSet, DataSet]:
    """Generates the DataSet object for the different splits of GSE73072."""

    # set data as dataframes
    train_data = pd.read_csv(train_data_path, index_col=0)
    test_data = pd.read_csv(test_data_path, index_col=0)

    # compute datasets
    ds_train = DataSet(data=train_data, metadata=metadata)
    ds_test = DataSet(data=test_data, metadata=metadata)

    # align train and test data on features
    train_pids = ds_train.data.columns
    test_pids = ds_test.data.columns 
    pids = train_pids.intersection(test_pids)
    ds_train = ds_train.slice_dataset(feature_ids=pids)
    ds_test = ds_test.slice_dataset(feature_ids=pids)

    # reset names
    ds_train.name = os.path.basename(train_data_path).rstrip('.csv')
    ds_test.name = os.path.basename(test_data_path).rstrip('.csv')

    return ds_train, ds_test 

def load_pathway_metadata(file_path: str) -> DataFrame:

    # load the metadata
    df = pd.read_csv(file_path, dtype=str, delimiter='\t')
    df.columns = ['EntrezID', 'ReactomeID', 'url', 'Name', 'TAS/EXP', 'Species']

    
    df = df.groupby('ReactomeID').agg(pd.unique)

    return df

def extract_pathway_genes(pathway_dir: str) -> pd.DataFrame:
    """
    Takes in a directory containing pathways files and
    returns a dataframe with the gene ids for each pathways along with the number
    of genes/probe_ids in a given pathway.
    """

    # init pathway information
    pathway_names = []
    n_pids = []
    for f in os.listdir(pathway_dir):
        # extract pathway name and number of probe ids
        start = f.find("R-HSA")
        end = f.find(".csv")
        pathway_names.append(f[start:end])
        pw_matrix = pd.read_csv(os.path.join(pathway_dir, f), index_col = 0)
        pids = list(pw_matrix.columns)
        n_pids.append(len(pids))

    # initialize output pathway genes
    gene_columns = ['n_probes'] + ['n_genes'] + [f"gid_{i}" for i in range(np.max(n_pids))]
    pathway_genes = pd.DataFrame(index=pathway_names, columns=gene_columns)
    pathway_edge_dir = '/data3/darpa/omics_databases/ensembl2pathway/pathways_edges/pathways/pw_edge_mtx'

    # find specific genes in each pathway
    for i, f in enumerate(os.listdir(pathway_edge_dir)):
        pw_matrix = pd.read_csv(os.path.join(pathway_edge_dir, f), index_col=0)
        genes = [g[7:] for g in list(pw_matrix.columns)]
        n_genes = len(genes)
        pathway_genes.iloc[i, 2:n_genes+2] = genes #maybe change this
        pathway_genes.iat[i, 0] = n_pids[i]
        pathway_genes.iat[i, 1] = n_genes 

    return pathway_genes

def reformat_pathway_metadata(pathway_dir: DataFrame, pathway_metadata: DataFrame) -> DataFrame:
    print(pathway_metadata)
    """Reformats pathway metadata so that it contains genes and gene counts and grouped by pathway id."""
    pathway_metadata_by_rid = pathway_metadata.groupby('ReactomeID').agg(pd.unique)
    pathway_genes = extract_pathway_genes(pathway_dir)
    pathway_info = pathway_metadata_by_rid[['Name', 'url']]
    pathway_metadata_new = pd.concat((pathway_info, pathway_genes), axis=1).loc[pathway_genes.index]
    pathway_metadata_new.index.name = 'ReactomeID'

    return pathway_metadata_new

def group_data_paths_by_experiment(data_file_paths: List[str]) -> List[Tuple[str, str, str]]:
    """Groups data paths for train and test data paths for each experiment. Each tuple is of
       of the form (experiment_name, train_data_path, test_data_path)"""

    # strip train/test from ends
    file_names = [os.path.basename(file_path).rstrip('.csv') for file_path in data_file_paths]

    # get experiment names
    experiment_names = np.array(['_'.join(file_name.split('_')[:-1]) for file_name in file_names])

    # get unique experiment names
    unique_experiments = np.unique(experiment_names)

    # initialize output
    out = []

    for experiment in unique_experiments:

        # find corresponding data files
        idx = np.where(experiment_names == experiment)[0]
        experiment_files = np.array(data_file_paths, dtype=str)[idx]

        # grab training data file
        train_file = [file for file in experiment_files if 'train' in file][0].item()
        test_file = [file for file in experiment_files if 'test' in file][0].item()

        # append to output
        out.append((experiment, train_file, test_file))

    return out
