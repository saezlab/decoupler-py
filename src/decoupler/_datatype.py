from typing import Union

import numpy as np
import pandas as pd
from anndata import AnnData

DataType = Union[
    AnnData,
    pd.DataFrame,
    tuple[np.ndarray, np.ndarray, np.ndarray],
]
