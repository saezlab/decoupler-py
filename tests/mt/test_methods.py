import pytest

import decoupler as dc


def test_methods():
    lstm = dc.mt._methods
    len_lstm = len(lstm)
    len_dfm = dc.mt.show().shape[0]
    assert len_lstm == len_dfm
    assert all(isinstance(m, dc._Method.Method) for m in lstm)
