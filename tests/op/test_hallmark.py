import pandas as pd
import numpy as np
import pytest

import decoupler as dc


def test_hallmark():
    hm = dc.op.hallmark()
    assert isinstance(hm, pd.DataFrame)
    cols = {'source', 'target'}
    assert cols.issubset(hm.columns)
    assert not hm.duplicated(['source', 'target']).any()
