import calcom
import pandas
import numpy as np

'''
We want to add the 6 to 1 experiment and all time bins

#H1N1 (both sets), H3N2/DEE5, HRV (both sets), and RSV

['gse73072_dee3', 'gse73072_dee4','gse73072_dee5', 'gse73072_duke', 'gse73072_uva', 'gse73072_dee1']

['gse73072_dee2']


code was taken from Flu_GSE73072_LOSO_Control_vs_Shedder_Timebin_Exp.py
same splits as those used in https://www.nature.com/articles/s41598-021-95293-z
'''

#StudyID 		disease
#gse73072_duke 	hrv
#gse73072_uva 	hrv
#gse73072_dee1 	rsv
#gse73072_dee3 	h1n1
#gse73072_dee4 	h1n1
#gse73072_dee2 	h3n2
#gse73072_dee5 	h3n2


def returnTimeid(timebin):

	l = int(timebin.split('_')[0])
	u = int(timebin.split('_')[1])
	if l < 0: #for controls
		return np.arange(l,u)
	else:
		return np.arange(l,u+1)

def loadCalcomDataUsingStudy(dataPath,timeBin,study):
	print('Loading controls and shedder from time bin',timeBin,'. Study: ',study)
	# create an ccd onject
	ccd = calcom.io.CCDataSet(dataPath)
	#quary string for controls
	q0 = {'time_id': returnTimeid('-100_1'), 'StudyID':study}

	#quary string for shedders
	q1 = {'time_id': returnTimeid(timeBin),'StudyID':study,'shedding':True}
	
	tmpString = 'Shedder'+timeBin

	ccd.generate_attr_from_queries('new_attr',{'Control':q0,tmpString:q1})	
	new_attr = {'control': q0, 'shedder'+timeBin: q1}
	classificationAttr='control-vs-sheddder'+timeBin
	ccd.generate_attr_from_queries(classificationAttr, new_attr)
	idxs = ccd.find(classificationAttr, ['control', 'shedder'+timeBin])
	
	return ccd, idxs, classificationAttr


def generate_dataset(dPath, tBin, study, train_or_test = None, prefix = None, limma_switch = ''):
    ccd, idx, classificationAttr = loadCalcomDataUsingStudy(dataPath=dPath,timeBin=tBin,study=study)
    
    data = ccd.generate_data_matrix(idx_list=idx)
    sample_ids = ccd.generate_labels('SampleID', idx_list=idx)
    

    if limma_switch == 'studyID':
        study_ids = ccd.generate_labels('StudyID', idx_list=idx)
        data = calcom.utils.limma(data, study_ids)
        was_limmad = f'_{limma_switch}_limma'
    elif limma_switch == 'subjectID':
        study_ids = ccd.generate_labels('SubjectID', idx_list=idx)
        data = calcom.utils.limma(data, study_ids)
        was_limmad = f'_{limma_switch}_limma'
    else:
        was_limmad = ''
    

    probe_ids = list(ccd.variable_names)
    all_data = pandas.DataFrame(data = data, columns = probe_ids, index = sample_ids)
    all_data.index.name = 'SampleID'

    # converter = pandas.read_csv('/data4/mankovic/GSE73072/probe_2_entrez_calcom.csv', index_col = 0)
    # all_data = all_data[converter.index]
    # converter_dict = converter.to_dict()
    # all_data = all_data.rename(columns = converter_dict['EntrezID'])
    #gse73072_4to2_25_32_limma_train

    base_dir = '../calcom_splits/data/'
    all_data.to_csv(f'{base_dir}gse73072_{prefix}_{tBin}{was_limmad}_{train_or_test}.csv')

    experiment_metadata = ccd.generate_metadata()[['SampleID', classificationAttr]]
    experiment_metadata = experiment_metadata.rename(columns = {classificationAttr: f'{prefix}_{tBin}_{train_or_test}'})
    experiment_metadata = experiment_metadata.set_index('SampleID')

    return experiment_metadata



if __name__ == "__main__":
    dPath = '/data3/darpa/all_CCD_processed_data/ccd_gse73072_original_microarray.h5'
    # tBin = '25_32'
    ccd = calcom.io.CCDataSet(dPath)
    metadata = ccd.generate_metadata()
    metadata = metadata.drop(columns = '_id')
    metadata = metadata.set_index('SampleID')

    for tBin in ['1_8','9_16','17_24', '25_32']:
        for lim_switch in ['subjectID', '']:
            study = ['gse73072_dee2', 'gse73072_dee3','gse73072_dee4', 'gse73072_dee5']
            four_three_train_metadata = generate_dataset(dPath, tBin, study, train_or_test = 'train', prefix = '4to3', limma_switch = lim_switch)

            study = ['gse73072_uva', 'gse73072_duke', 'gse73072_dee1']
            four_three_test_metadata = generate_dataset(dPath, tBin, study, train_or_test = 'test', prefix = '4to3', limma_switch = lim_switch)

            study = ['gse73072_dee2', 'gse73072_dee3','gse73072_dee4', 'gse73072_dee5']
            four_two_train_metadata = generate_dataset(dPath, tBin, study, train_or_test = 'train', prefix = '4to2', limma_switch = lim_switch)

            study = ['gse73072_uva', 'gse73072_duke']
            four_two_test_metadata = generate_dataset(dPath, tBin, study, train_or_test = 'test', prefix = '4to2', limma_switch = lim_switch)

            study = ['gse73072_dee3', 'gse73072_dee4','gse73072_dee5', 'gse73072_duke', 'gse73072_uva', 'gse73072_dee1']
            six_one_train_metadata = generate_dataset(dPath, tBin, study, train_or_test = 'train', prefix = '6to1', limma_switch = lim_switch)

            study = ['gse73072_dee2']
            six_one_test_metadata = generate_dataset(dPath, tBin, study, train_or_test = 'test', prefix = '6to1', limma_switch = lim_switch)


            # study = ['gse73072_duke', 'gse73072_uva', 'gse73072_dee3','gse73072_dee4', 'gse73072_dee5']
            # five_one_train_metadata = generate_dataset(dPath, tBin, study, train_or_test = 'train', prefix = '5to1', limma_switch = lim_switch)

            if lim_switch == '':
                # study = ['gse73072_dee2']
                # five_one_test_metadata = generate_dataset(dPath, tBin, study, train_or_test = 'test', prefix = '5to1', limma_switch = lim_switch)
                #the dataset with limma is just a copy of this dataset
                
                metadata = pandas.concat((metadata, four_three_train_metadata), axis=1).loc[metadata.index]
                metadata = pandas.concat((metadata, four_three_test_metadata), axis=1).loc[metadata.index]

                metadata = pandas.concat((metadata, four_two_train_metadata), axis=1).loc[metadata.index]
                metadata = pandas.concat((metadata, four_two_test_metadata), axis=1).loc[metadata.index]

                metadata = pandas.concat((metadata, six_one_train_metadata), axis=1).loc[metadata.index]
                metadata = pandas.concat((metadata, six_one_test_metadata), axis=1).loc[metadata.index]

                # metadata = pandas.concat((metadata, five_one_train_metadata), axis=1).loc[metadata.index]
                # metadata = pandas.concat((metadata, five_one_test_metadata), axis=1).loc[metadata.index]
    metadata.to_csv('../calcom_splits/gse73072_metadata.csv')

