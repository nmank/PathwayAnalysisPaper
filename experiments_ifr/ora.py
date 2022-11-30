import pandas as pd
import reactome2py
from reactome2py import analysis
import os

base_dir = '/data4/mankovic/GSE73072/features'
out_dir = '/data4/mankovic/GSE73072/ge_ora'

for f in os.listdir(base_dir):
    print(f'experiment: {f[:-4]}')

    file_name = os.path.join(base_dir, f)

    featureset = pd.read_csv(file_name, index_col = 0)

    probes = ','.join(list(featureset['ProbeID']))

    reactome_results = reactome2py.analysis.identifiers(ids=probes, interactors=False, page_size='2000', page='1', species='Homo Sapiens', sort_by='ENTITIES_FDR', order='ASC', resource='TOTAL', p_value='1', include_disease=True, min_entities=None, max_entities=None, projection=False)

    ora_df = pd.DataFrame(columns = ['ID', 'pvalue'])

    n_pathways = 0
    for p in reactome_results['pathways']:
        row = pd.DataFrame(data = [[p['stId'], p['entities']['pValue']]], columns = ['ID', 'pvalue'])
        ora_df = ora_df.append(row, ignore_index = True)
        n_pathways += 1

    out_file = os.path.join(out_dir, f)
    ora_df.to_csv(out_file)

    print(f'{n_pathways} pathways (2000 max)')