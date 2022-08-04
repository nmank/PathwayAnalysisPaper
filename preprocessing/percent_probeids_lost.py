import numpy as np
import pandas as pandas
import os

probe_conversion = pandas.read_csv('../probe_2_entrez_calcom_v2.csv')

data = pandas.read_csv('../calcom_splits/data/gse73072_4to3_1_8_train.csv')

n_probe_ids = len(np.unique(probe_conversion['ProbeID']))

conversion_eids = [str(e) for e in np.unique(probe_conversion['EntrezID'])]
n_entrez_ids = len(conversion_eids)


eids = set()
for f in os.listdir('../pathways_edges/pathways/pw_edge_mtx/'):

    f_path = '../pathways_edges/pathways/pw_edge_mtx/'+f

    p_data = pandas.read_csv(f_path, index_col = 0)

    eids = eids.union(set([e[7:] for e in p_data.columns]))

print(f'{len(eids)} eids in pathway matrices')
print(f'{n_entrez_ids} eids in probe2entrez file')
print(f'intersection of entrezIDs is {len(set(conversion_eids).intersection(eids))}')

print(f'{np.round(1-n_probe_ids/ len(data[1:].columns),3)*100}% of probe ids are lost')