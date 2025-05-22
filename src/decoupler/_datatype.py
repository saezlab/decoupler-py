from typing import Union, Tuple

from anndata import AnnData
import pandas as pd
import numpy as np


DataType = Union[
    AnnData,
    pd.DataFrame,
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]
