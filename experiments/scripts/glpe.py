"""This script performs centrality ranking of pathways based on a set of selected features."""

#use this:   mlflow run . --entry-point glpe --no-conda --experiment-id 3

#nate added this import:
import sys

sys.path.append('/data4/mankovic/GSE73072/experiments/')

# imports
from matplotlib import use
import mlflow
import sys
import argparse
import glob
import os
import itertools
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series
from typing import List, Tuple
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

sys.path.append('/home/katrina/a/mankovic/')
from PathwayAnalysis.GLPE import CLPE, LPE
import PathwayAnalysis

from orthrus.core.pipeline import *
from orthrus.sparse.classifiers.svm import SSVMSelect
from sklearn.svm import LinearSVC as SVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from modules.utils import generate_datasets, load_pathway_metadata, \
group_data_paths_by_experiment, reformat_pathway_metadata
from modules.logging import log_confusion_stats

# set command line params
parser = argparse.ArgumentParser(description="This script performs pathway selection using SSVM on " \
                                             "pathway expression transformed data and then tests " \
                                             "on sequestered test data.")
parser.add_argument("--data-dir", dest="data_dir", type=str,
                    help="The directory containing the normalized data files.")
parser.add_argument("--metadata-path", dest="metadata_path", type=str,
                    help="The file containing the metadata for the samples.")
parser.add_argument("--pathway-dir", dest="pathway_dir", type=str,
                    help="The directory containing the pathway csv files.")
parser.add_argument("--pathway-metadata", dest="pathway_metadata", type=str,
                    help="The csv file containing metadata for each pathway.")
parser.add_argument("--debug", dest="debug", action='store_true', help="Debug flag.")
parser.set_defaults(debug=False)
args = parser.parse_args()

# script specific functions
def generate_glpe_transform(pathway_metadata: DataFrame,
                            pathway_dir: str,
                            feature_ids: Series, similarity_measure: str,
                            centrality_measure: str, directed: bool) -> Transform:
    """Generates a Transform object for GLPE to be used in an orthrus pipeline."""

    if centrality_measure is None:
        # generate LPE instance
        glpe = LPE(feature_ids=feature_ids, #list of the probeids or entrez ids for the dataset 
                   pathway_files=pathway_dir, #probeid networks
                   normalize_rows=True )
    else:
        # generate CLPE instance
        glpe = CLPE(centrality_measure = centrality_measure, 
                network_type=similarity_measure,
                feature_ids=feature_ids,
                pathway_files=pathway_dir,
                directed=directed,
                heat_kernel_param=2,
                normalize_rows=True)

    # create transform process
    transform = Transform(process=glpe,
                          process_name='glpe',
                          new_f_ids=glpe.pathway_names,
                          vardata=pathway_metadata,
                          )

    return transform

def generate_pathway_selection_pipeline(glpe: Transform, class_attr: str) -> Pipeline:
    """Generates an orthrus pipeline for pathway selection using SSVM given a GLPE transformation."""

    # generate standardization
    std = Transform(process=StandardScaler(),
                    process_name='std',
                    retain_f_ids=True,
                    )

    # ssvm feature selection
    ssvm = FeatureSelect(process=SSVMSelect(corr_threshold=.90,
                                            show_plot=False,
                                            n_features=2261),
                         process_name='ssvmc',
                         supervised_attr=class_attr,
                         f_ranks_handle='f_ranks',
                         )

    # create pipeline
    pipeline = Pipeline(processes=(std, glpe, std, ssvm),
                        pipeline_name='glpe_select',
                        verbosity=2,
                        )

    return pipeline  

def compute_pathway_expression(selection_pipeline: Pipeline, ds: DataSet, use_selected: bool=True) -> DataSet:
    """Generates a down-selected pathway expression dataset from a fitted pathway selection pipeline and a new dataset."""

    # make it so pathway selector does not refit
    selector = selection_pipeline.processes[-1].collapse_results()['selector']['batch']
    ssvm = FeatureSelect(process=selector,
                         process_name='ssvm',
                         prefit=True)

    # make it so glpe does not refit
    glpe = selection_pipeline.processes[1]
    transformer = glpe.collapse_results()['transformer']['batch']
    glpe = Transform(process=transformer,
                     process_name='glpe',
                     vardata=glpe.vardata,
                     new_f_ids=glpe.process.pathway_names,
                     prefit=True)

    # generate standardization
    std = Transform(process=StandardScaler(),
                    process_name='std',
                    retain_f_ids=True,
                    )

    # create pipeline
    if use_selected:
        pipeline = Pipeline(processes=(std, glpe, std, ssvm),
                            pipeline_name='pathway_expression',
                            verbosity=2,
                            )
    else:
        pipeline = Pipeline(processes=(std, glpe, std),
                            pipeline_name='pathway_expression',
                            verbosity=2,
                            )

    
    # run the pipeline
    pipeline.run(ds)
    transform = pipeline.results_['batch']['transform']
    
    # return the new dataset
    return transform(ds)

def generate_pathway_classification_pipeline(class_attr: str) -> Pipeline:
    """Generates an orthrus pipeline for classification using SSVM."""
    list_of_class = class_attr.split('_')

    # leave-one-out
    logo = Partition(process=LeaveOneGroupOut(),
                     process_name='logo',
                     split_group='SubjectID',
                     )

    # svm classification
    svm = Classify(process=SVM(max_iter = 20000),
                   process_name="svm",
                   class_attr=class_attr,
                   f_weights_handle="coef_",
                   )

    # scoring
    # report = Report(pred_attr="shedding",
    #                 )
    conf_mat = Score(process=confusion_matrix,
                     pred_attr=class_attr,
                     process_name='conf_mat',
                     classes=[f'shedder{list_of_class[1]}_{list_of_class[2]}', 'control']
                     )

    # create pipeline
    pipeline = Pipeline(processes=(logo, svm, conf_mat),
                        pipeline_name='classify',
                        verbosity=2,
                        )

    return pipeline  

def log_pathways(selection_pipeline: Pipeline, pathway_metadata: DataFrame=None):
    """logs the csv of pathway ranks from a pathway selection pipeline."""

    # extract the ranked pathways from the pipeline
    selector: FeatureSelect = selection_pipeline.processes[-1]
    ranks = selector.collapse_results()['f_ranks']

    if pathway_metadata is not None:
        # concatenate with pathway metadata
        ranks = pd.concat((ranks, pathway_metadata), axis=1)

    # compute number of features selected
    mlflow.log_metric('n_features', ranks['batch:Selected'].sum())
    
    # save artifact
    temp_path = 'temp/pathway_ranks.csv'
    ranks.to_csv(temp_path)
    mlflow.log_artifact(temp_path)

def label_directed(tf: bool):
    """Gives human readable label to boolean directed label."""

    if tf:
        return "directed"
    else:
        return "undirected"

def compute_class_attr(dataset_name: str):
    """Compute the classification attribute from an experiment."""
    #nate changed this
    # separate experiment name
    exp_name_sep = dataset_name.split('_')

    # # extract info
    train_test_type = exp_name_sep[1]
    time_bin = f'{exp_name_sep[2]}_{exp_name_sep[3]}'
    split_type = exp_name_sep[-1]

    return f"{train_test_type}_{time_bin}_{split_type}"
  
def visualize(ds: DataSet, save_name: str, attr: str):

    # use pca
    pca = PCA(n_components=2, whiten=True)
    ds.path = 'temp'

    ds.visualize(embedding=pca,
                attr=attr,
                cross_attr='time_id',
                title='',
                subtitle='',
                palette='bright',
                alpha=.75,
                save=True,
                save_name=save_name)


    # log with mlflow
    mlflow.log_artifact(os.path.join('temp', save_name + '.png'))

# main function
def main() -> None:
    # debug
    if args.debug:
        os.environ["MLFLOW_EXPERIMENT_ID"] = '1'

    # get file paths
    data_file_paths = glob.glob(f"{args.data_dir}/*.csv")
    pathway_dir = args.pathway_dir

    # load metadata
    metadata = pd.read_csv(args.metadata_path, index_col=0)

    # generate train/test paths by experiment
    train_test_paths = group_data_paths_by_experiment(data_file_paths)

    # load/reformat the pathway metadata
    pathway_metadata = load_pathway_metadata(args.pathway_metadata)
    pathway_metadata = reformat_pathway_metadata(pathway_dir, pathway_metadata)

    # set similarity and centrality measures
    similarity_measures = ['precomputed', 'correlation']
    centrality_measures = ['degree', 'page_rank', None]

    for experiment, train_path, test_path in train_test_paths:
        # generate train/test datasets
        ds_train, ds_test = generate_datasets(train_path, test_path, metadata)

        # set classification attributes
        class_attr_train = compute_class_attr(ds_train.name)
        class_attr_test = compute_class_attr(ds_test.name)

        # loop through different methods
        param_combinations = itertools.product(similarity_measures,
                                               centrality_measures,
                                               [True, False],
                                               )

        for similarity_measure, centrality_measure, directed in param_combinations:

            # generate run id
            run_name = f"{experiment}_{centrality_measure}_{similarity_measure}_{label_directed(directed)}"

            # start an mlflow run
            with mlflow.start_run(experiment_id=os.environ['MLFLOW_EXPERIMENT_ID']):

                    # visualize gene expression
                    visualize(ds_train, 'train_ge', class_attr_train)
                    visualize(ds_test, 'test_ge', class_attr_test)

                    # log the parameters
                    mlflow.set_tag('mlflow.runName', run_name)
                    mlflow.log_param('experiment', experiment)
                    mlflow.log_param('train_data_path', train_path)
                    mlflow.log_param('test_data_path', test_path)
                    mlflow.log_param('similarity_measure', similarity_measure)
                    mlflow.log_param('centrality_measure', centrality_measure)
                    mlflow.log_param('directed', str(directed))
                    if centrality_measure is None:
                        mlflow.log_param('method', 'LPE')
                    else:
                        mlflow.log_param('method', 'CPE')
                
                    # generate glpe
                    glpe = generate_glpe_transform(pathway_metadata=pathway_metadata,
                                                   pathway_dir=pathway_dir,
                                                   feature_ids=ds_train.data.columns,
                                                   similarity_measure=similarity_measure,
                                                   centrality_measure=centrality_measure,
                                                   directed=directed)

                    # generate selection pipeline
                    selection_pipeline = generate_pathway_selection_pipeline(glpe=glpe, class_attr=class_attr_train)

                    # fit to training data
                    selection_pipeline.run(ds_train)
                        
                    ds_train_pe_selected = compute_pathway_expression(selection_pipeline, ds_train)
                    visualize(ds_train_pe_selected, 'train_pe_selected', class_attr_train)

                    # compute pathway expression using all pathways
                    ds_train_pe = compute_pathway_expression(selection_pipeline, ds_train, use_selected=False)
                    visualize(ds_train_pe, 'train_pe', class_attr_train)

                    # log the pathways
                    log_pathways(selection_pipeline, pathway_metadata)

                    # visualize test data pathway expression on all pathways
                    ds_test_pe = compute_pathway_expression(selection_pipeline, ds_test, use_selected=False)
                    visualize(ds_test_pe, 'test_pe', class_attr_test)

                    # compute pathway expression on new data
                    ds_test_pe = compute_pathway_expression(selection_pipeline, ds_test)
                    visualize(ds_test_pe, 'test_pe_selected', class_attr_test)

                    #nate changed this to save pathway expression test data
                    ds_test_pe.save(file_path = f'temp/PE_test.ds', overwrite = True)
                    # log with mlflow
                    mlflow.log_artifact('temp/PE_test.ds')

                    # send the pathways and clpe to the classification pipeline
                    classification_pipeline = generate_pathway_classification_pipeline(class_attr=class_attr_test)

                    # run classification pipeline on test data
                    classification_pipeline.run(ds_test_pe)

                    # log classification metrics
                    conf_mat: Score = classification_pipeline.processes[-1]
                    log_confusion_stats(conf_mat)

if __name__=="__main__":

    # run main
    main()
