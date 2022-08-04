# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 15:37:46 2022

@author: csu_l
"""

import pandas as pd
import numpy as np
from os import listdir
import glob

pathway_data = pd.read_csv('/data3/darpa/omics_databases/ensembl2pathway/pathways_edges/pathways/NCBI2Reactome_All_Levels.txt', delimiter = '\t', header = None)[[0,1]]

#need to remove duplicates. Long story one of the later columns will have a different value but it doesn't matter
pathway_data=pathway_data.drop_duplicates()

pathway_data.columns = ['eid','pid']

#for the sake of  not doing a loop you might consider groupby. It is equivalent(ish) to R's group_by I use a lot

pw_ct=pathway_data.groupby('pid').count()


#also we just need the human ones. 
pw_ct=pw_ct.reset_index()
pid_col=pw_ct['pid']
pw_ct_human=pw_ct[pid_col.str.startswith('R-HSA')]



pw_ct_human.to_csv('/data3/darpa/omics_databases/ensembl2pathway/pathway_sizes_human.csv')

###here is the edge matrix creations

files=glob.glob("/data3/darpa/omics_databases/ensembl2pathway/pathways_edges/pathways/pw_edge_mtx/*.csv")

dict1={}
for i in files:
    pid= i.replace("/data3/darpa/omics_databases/ensembl2pathway/pathways_edges/pathways/pw_edge_mtx/pw_mtx_", '').replace('.csv', '')
    df=pd.read_csv(i)
    pw_size=len(df)
    dict1[pid]=pw_size
    
df_pw_from_mtx=pd.DataFrame.from_dict([dict1])
#bad policy 

df_pw_from_mtx.to_csv("/data3/darpa/omics_databases/ensembl2pathway/pw_sizes_from_mtx.csv")

#####now comparing the two 

df_mtx=df_pw_from_mtx.T
df_mtx.columns = ['genes_mtx']

#df_mtx=df_mtx.set_index("pid")
pw_ct_human=pw_ct_human.set_index("pid")

df_inner=df_mtx.join(pw_ct_human, how="inner", lsuffix="_edge", rsuffix="_default")

df_inner["wrong0right1"]= np.where(df_inner["genes_mtx"]==df_inner["eid"], 1, 0)
df_inner.to_csv("/data3/darpa/omics_databases/ensembl2pathway/pw_sizes_compare1.csv")