import pandas
import os


for f in os.listdir('../pathways_edges/pathways/pw_edge_mtx/'):

    f_path = '../pathways_edges/pathways/pw_edge_mtx/'+f

    p_data = pandas.read_csv(f_path, index_col = 0)

    x = pandas.read_csv('../probe_2_entrez_calcom_v2.csv')
    translate_dict = {}
    for e in list(x['EntrezID']):
        translate_dict[str(e)] = list(x.query(f"EntrezID == {e}")['ProbeID'])


    translate_eids = {}
    for c in p_data.columns:
        translate_eids[c] = str(c[7:])

    p_data = p_data.rename(columns = translate_eids)
    p_data = p_data.rename(index = translate_eids)

    eids = list(set(p_data.columns).intersection(set(translate_dict.keys())))

    p_data = p_data[eids].loc[eids]

    new_pids = []
    for e in eids:
        new_pids += translate_dict[e]

    new_p_data = pandas.DataFrame(index = eids, columns = new_pids)
    for e in eids:
        for pid in translate_dict[e]:
            new_p_data[pid] = p_data[e]

    more_new_p_data = pandas.DataFrame(index = new_pids, columns = new_pids)
    for e in eids:
        for pid in translate_dict[e]:
            more_new_p_data.loc[pid] = new_p_data.loc[e]
    
    more_new_p_data.to_csv('../new_pw_edge_mtx_v2/'+f)

