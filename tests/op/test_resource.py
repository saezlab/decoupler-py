import pandas as pd
import numpy as np
import pytest

import decoupler as dc


def test_show_resources():
    df = dc.op.show_resources()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert {'name', 'license'}.issubset(df.columns)
    assert np.isin(['PROGENy', 'MSigDB'], df['name']).all()


@pytest.mark.parametrize('name', ['Lambert2018', 'PanglaoDB'])
def test_resource(
    name
):
    rs = dc.op.resource(name=name)
    assert isinstance(rs, pd.DataFrame)
    assert 'genesymbol' in rs.columns
