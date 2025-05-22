import pandas as pd
import pytest

import decoupler as dc

    
def test_infer_dtypes():
    df = pd.DataFrame(
        data = [
            ['1', 'A', 'true', 'False', 0.3],
            ['2', 'B', 'false', 'True', 0.1],
            ['3', 'C', 'false', 'True', 3.1],
        ],
        columns=['a', 'b', 'c', 'd', 'e'],
        index=[0, 1, 2],
    )
    df['b'] = df['b'].astype('string')
    idf = dc.op._dtype._infer_dtypes(df.copy())
    assert pd.api.types.is_numeric_dtype(idf['a'])
    assert idf['b'].dtype == 'object'
    assert pd.api.types.is_bool_dtype(idf['c'])
    assert pd.api.types.is_bool_dtype(idf['d'])
    assert pd.api.types.is_numeric_dtype(idf['e'])

    