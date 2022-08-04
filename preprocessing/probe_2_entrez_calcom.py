import calcom
import pandas
import numpy as np
import sys
import platform_lookup
import os

def probe2entrez():
    '''
    take the first entrez id for each probe id
    '''

    eids = set()
    for f in os.listdir('../pathways_edges/pathways/pw_edge_mtx/'):

        f_path = '../pathways_edges/pathways/pw_edge_mtx/'+f

        p_data = pandas.read_csv(f_path, index_col = 0)

        eids = eids.union(set([e[7:] for e in p_data.columns]))

    pl = platform_lookup.platform_lookup(datafile='../platform/GPL571-17391_DATA.txt',
                                        metafile='../platform/GPL571-17391_META.txt')
                                        
    ccd = calcom.io.CCDataSet('/data3/darpa/all_CCD_processed_data/ccd_gse73072_original_microarray.h5')
    probe_ids = list(ccd.variable_names)
    print(len(probe_ids))
    it_worked, entrez_ids = pl.convert(probe_ids, ref_set = list(eids), rule = 'overlap_first')
    probe_ids = np.array(probe_ids)

    print(len(probe_ids[it_worked]))
    conversion_df = pandas.DataFrame(columns = ['ProbeID', 'EntrezID'])

    conversion_df['ProbeID'] = probe_ids[it_worked]
    conversion_df['EntrezID'] = entrez_ids[it_worked]

    conversion_df = conversion_df.set_index('ProbeID')
    
    return conversion_df
    

probe_2_entrez_df = probe2entrez()

probe_2_entrez_df.to_csv('../probe_2_entrez_calcom_v2.csv')