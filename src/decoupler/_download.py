import io

import pandas as pd
import requests
from tqdm import tqdm

from decoupler._log import _log

URL_DBS = "https://omnipathdb.org/annotations?databases="
URL_INT = "https://omnipathdb.org/interactions/?genesymbols=1&"


def _download(
    url: str,
    verbose: bool = False,
) -> io.BytesIO:
    assert isinstance(url, str), "url must be str"
    # Download with progress bar
    m = f"Downloading {url}"
    _log(m, level="info", verbose=verbose)
    chunks = []
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with tqdm(unit="B", unit_scale=True, desc="Progress", disable=not verbose) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    chunks.append(chunk)
                    pbar.update(len(chunk))
    # Read into bytes
    data = io.BytesIO(b"".join(chunks))
    m = "Download finished"
    _log(m, level="info", verbose=verbose)
    return data


def _bytes_to_pandas(data: io.BytesIO, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(data, **kwargs)
    return df
