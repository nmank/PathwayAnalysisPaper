"""This script performs centrality ranking of pathways based on a set of selected features."""

#use this:   mlflow run . --entry-point svm --no-conda --experiment-id 1

#nate added this import:
# import sys
# sys.path.append('/home/katrina/a/mankovic/')
# sys.path.append('/data4/mankovic/GSE73072/experiments/')

# imports
import mlflow
import argparse
import glob
import os
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
from orthrus.core.pipeline import *
from orthrus.sparse.classifiers.svm import SSVMSelect
from sklearn.svm import LinearSVC as SVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.decomposition import PCA
from modules.utils import generate_datasets, group_data_paths_by_experiment
from modules.logging import log_confusion_stats

# set command line params
parser = argparse.ArgumentParser(description="This script performs SSVM on training data to extract features" \
                                             "and then tests the features on sequestered test data with SVM.")
parser.add_argument("--data-dir", dest="data_dir", type=str,
                    help="The directory containing the normalized data files.")
parser.add_argument("--metadata-path", dest="metadata_path", type=str,
                    help="The file containing the metadata for the samples.")
parser.add_argument("--debug", dest="debug", action='store_true', help="Debug flag.")
parser.set_defaults(debug=False)
args = parser.parse_args()

# script specific functions
def generate_feature_selection_pipeline(class_attr: str) -> Pipeline:
    """Generates an orthrus pipeline for pathway selection using SSVM given a GLPE transformation."""

    # generate standardization
    std = Transform(process=StandardScaler(),
                    process_name='std',
                    retain_f_ids=True,
                    )

    # ssvm feature selection
    ssvm = FeatureSelect(process=SSVMSelect(corr_threshold=.90,
                                            show_plot=False),
                         process_name='ssvmc',
                         supervised_attr=class_attr,
                         f_ranks_handle='f_ranks',
                         )

    # create pipeline
    pipeline = Pipeline(processes=(std, ssvm),
                        pipeline_name='ssvm_select',
                        verbosity=2,
                        )

    return pipeline  

def compute_feature_expression(selection_pipeline: Pipeline, ds: DataSet) -> DataSet:
    """Generates a down-selected feature expression dataset from a fitted feature selection pipeline and a new dataset."""

    # make it so pathway selector does not refit
    selector = selection_pipeline.processes[-1].collapse_results()['selector']['batch']
    ssvm = FeatureSelect(process=selector,
                         process_name='ssvm',
                         prefit=True)

    # generate standardization
    std = Transform(process=StandardScaler(),
                    process_name='std',
                    retain_f_ids=True,
                    )

    # create pipeline
    pipeline = Pipeline(processes=(std, ssvm),
                        pipeline_name='feature_expression',
                        verbosity=2,
                        )
    
    # run the pipeline
    pipeline.run(ds)
    transform = pipeline.results_['batch']['transform']
    
    # return the new dataset
    return transform(ds)

def generate_classification_pipeline(class_attr: str) -> Pipeline:
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

def log_features(selection_pipeline: Pipeline, feature_metadata: DataFrame=None):
    """logs the csv of pathway ranks from a pathway selection pipeline."""

    # extract the ranked pathways from the pipeline
    selector: FeatureSelect = selection_pipeline.processes[-1]
    ranks = selector.collapse_results()['f_ranks']

    if feature_metadata is not None:
        # concatenate with pathway metadata
        ranks = pd.concat((ranks, feature_metadata), axis=1)
    
    # compute number of features selected
    mlflow.log_metric('n_features', ranks['batch:Selected'].sum())

    # save artifact
    temp_path = 'temp/feature_ranks.csv'
    ranks.to_csv(temp_path)
    mlflow.log_artifact(temp_path)

#def compute_class_attr(dataset_name: str):
#    """Compute the classification attribute from an experiment."""
#    
#    # separate experiment name
#    exp_name_sep = dataset_name.split('_')
#
#    # extract info
#    train_test_type = exp_name_sep[1]
#    split_type = exp_name_sep[-1]
#
#    return f"{train_test_type}_{split_type}"

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

    # load metadata
    metadata = pd.read_csv(args.metadata_path, index_col=0)

    # generate train/test paths by experiment
    train_test_paths = group_data_paths_by_experiment(data_file_paths)

    for experiment, train_path, test_path in train_test_paths:
        # generate train/test datasets
        ds_train, ds_test = generate_datasets(train_path, test_path, metadata)

        # set classification attributes
        class_attr_train = compute_class_attr(ds_train.name)
        class_attr_test = compute_class_attr(ds_test.name)

        # generate run id
        run_name = f"{experiment}_ssvm"

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
            mlflow.log_param('method', 'GE')

            # generate selection pipeline
            selection_pipeline = generate_feature_selection_pipeline(class_attr=class_attr_train)

            # fit to training data
            selection_pipeline.run(ds_train)

            # log the features
            log_features(selection_pipeline, ds_train.vardata)

            ds_train_selected = compute_feature_expression(selection_pipeline, ds_train)
            visualize(ds_train_selected, 'train_ge_selected', class_attr_train)

            # compute pathway expression on new data
            ds_test_selected = compute_feature_expression(selection_pipeline, ds_test)
            visualize(ds_test_selected, 'test_ge_selected', class_attr_test)

            # send the pathways and clpe to the classification pipeline
            classification_pipeline = generate_classification_pipeline(class_attr=class_attr_test)

            # run classification pipeline on test data
            classification_pipeline.run(ds_test_selected)

            # log classification metrics
            conf_mat: Score = classification_pipeline.processes[-1]
            log_confusion_stats(conf_mat)

if __name__=="__main__":

    # run main
    main()
