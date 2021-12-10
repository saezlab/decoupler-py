import numpy as np
import pandas as pd

def m_rename(m, name):
    # Rename
    m = m.rename({'index':'sample', 'variable':'source'}, axis=1)

    # Assign score or pval
    if 'pval' in name:
        m = m.rename({'value':'pval'}, axis=1)
    else:
        m = m.rename({'value':'score'}, axis=1)
    
    return m


def melt(df):
    """
    Function to generate a long format dataframe similar to the one obtianed in
    the R implementation of decoupleR.
    
    Parameters
    ----------
    df : dict, tuple, list or pd.DataFrame
        Output of decouple, of an individual method or an individual dataframe.
    
    Returns
    -------
    m : melted long format dataframe.
    """
    
    # If input is result from decoule function
    if type(df) is list or type(df) is tuple:
        df = {k.name:k for k in df}
    if type(df) is dict:
        # Get methods run
        methods = np.unique([k.split('_')[0] for k in df])
        
        res = []
        for methd in methods:
            for k in df:
                # Extract pvals from this method
                pvals = df[methd+'_pvals'].reset_index().melt(id_vars='index')['value'].values
                
                # Melt estimates
                if methd in k and 'pvals' not in k:
                    m = df[k]
                    name = m.name
                    
                    m = m.reset_index().melt(id_vars='index')
                    
                    m = m_rename(m, name)
                    m['method'] = methd
                    m['pval'] = pvals
                    
                    res.append(m)
        
        # Concat results
        m = pd.concat(res)
            
    # If input is an individual dataframe
    elif type(df) is pd.DataFrame:
        # Melt
        name = df.name
        m = df.reset_index().melt(id_vars='index')
        
        # Rename
        m = m_rename(m, name)
    
    else:
        raise ValueError('Input type {0} not supported.'.format(type(df)))

    return m
