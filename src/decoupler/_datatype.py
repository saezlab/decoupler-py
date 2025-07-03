import numpy as np
import pandas as pd
from anndata import AnnData

DataType = AnnData | pd.DataFrame | tuple[np.ndarray, np.ndarray, np.ndarray]
