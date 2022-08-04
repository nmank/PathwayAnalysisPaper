class platform_lookup:
    def __init__(self, 
        datafile='/data3/darpa/omics_databases/platform/GPL571-17391_DATA.txt',
        metafile='/data3/darpa/omics_databases/platform/GPL571-17391_META.txt'
        ):
        import pandas
        
        self.df = pandas.read_csv(datafile, delimiter='\t')
        self.df.columns = [s.upper() for s in self.df.columns]
        # todo - get better. do we really care about this anyway?
        try:
            f = open(metafile,'r')
            self.meta = f.readlines()
            f.close()
        except:
            self.meta = 'No metadata available'
        #
        return
    #
    def lookup(self,probe,returns='ENTREZ_GENE_ID'):
        '''
        return another entry in dataframe associated with requested probe.
        Default: Entrez Gene ID
        
        Assumes the dataframe associated with the platform already loaded.
        '''
        import numpy as np
        probes = self.df['ID'].values
        idx = np.where(probes==probe)[0]
        if len(idx)==0:
            raise Exception('No %s conversion found for probe %s.'%(returns,probe))
        #
        
        val = self.df.iloc[idx[0]][returns]
        
        if isinstance(val, float):
            if np.isnan(val):
                raise Exception('No %s conversion found for probe %s.'%(returns,probe))
            else:
                # pandas reading integer values as floats; correct.
                val = str(int(val))
        elif isinstance(val,int):
            val = str(val)
        #
        return val
    #
    
    def convert(self, probe_list, returns='ENTREZ_GENE_ID', rule='first', verbosity=1):
        '''
        Applies lookup to each entry in probe_list. If there are multiple entries,
        denoted by three slashes ///, it takes the first entry by convention.
        
        NO INPUT CHECKING IS DONE. WE'RE EXPECTING AN ARRAY-LIKE OF STRINGS.
        
        Returns: 
            (1) A True/False mask; False values failed to find a conversion;
            (2) An array of Entrez Gene IDs. There may be duplicates.
            
            Entries with no conversion are labeled "FAILED".
        '''
        import numpy as np
        
        # make copy to be overwritten.
        end_list = np.array(probe_list)
        mask = np.ones(len(probe_list), dtype=bool)
        
        for j,p in enumerate(probe_list):
            try:
                out = self.lookup(p, returns=returns)
                out = out.split('///')
                if rule=='first':
                    out = out[0].strip()
                else:
                    # TODO: something else; for now do it anyway.
                    out = out[0].strip()
                #
                end_list[j] = out
            except:
                end_list[j] = 'FAILED'
                mask[j] = False
            #
            if verbosity>0:
                if (j)%(len(probe_list)//10)==0 or j==len(probe_list)-1:
                    print('%i of %i'%(j+1,len(probe_list)))
        #
        return mask, end_list
    #
#

if __name__=="__main__":
    pl = platform_lookup()  # defaults to GSE61754 for now.
    name0 = pl.lookup('1007_s_at')    # success
    print('1007_s_at -> %s'%name0)
    
    try:
        name1 = pl.lookup('garfield')   # failure; won't be in list
    except:
        print('failed to find probe garfield')
    #
    
    try:
        name2 = pl.lookup('AFFX-TrpnX-3_at')   # failure; has no Entrez Gene ID.
    except:
        print('failed to find an Entrez Gene ID for AFFX-TrpnX-3_at')
    #
    
