import pandas as pd
import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'url,kwargs',
    [
        [
            ('https://www.ncbi.nlm.nih.gov/geo/download/?' +
             'acc=GSM8563697&format=file&file=GSM8563697%' +
             '5FCO37%5Ffeatures%2Etsv%2Egz'),
            dict(sep='\t', compression='gzip', header=None)
        ],
        [
            ('https://www.ncbi.nlm.nih.gov/geo/download/?' +
             'acc=GSM8563697&format=file&file=GSM8563697%' +
             '5FCO37%5Ftissue%5Fpositions%5Flist%2Ecsv%2Egz'),
            dict(sep=',', compression='gzip')
        ], 
    ]
)
def test_download(
    url,
    kwargs,
):
    df = dc._download._download(url, **kwargs)
    assert isinstance(df, pd.DataFrame)
    assert df.columns.size > 1
