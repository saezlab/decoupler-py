import io
import time
from importlib.metadata import version

import pandas as pd
import requests
from tqdm import tqdm

from decoupler._log import _log

URL_DBS = "https://omnipathdb.org/annotations?databases="
URL_INT = "https://omnipathdb.org/interactions/?genesymbols=1&"


def _download_chunks(
    url: str,
    verbose: bool = False,
) -> io.BytesIO:
    assert isinstance(url, str), "url must be str"
    # Download with progress bar
    chunks = []
    __version__ = version("decoupler")
    headers = {"User-Agent": f"decoupler/{__version__} (https://github.com/scverse/decoupler)"}
    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        total = r.headers.get("Content-Length")
        total = int(total) if total and total.isdigit() else None
        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Progress",
            disable=not verbose,
            bar_format="{l_bar}{bar:20}{r_bar}",
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if not chunk:
                    continue
                chunks.append(chunk)
                pbar.update(len(chunk))
    # Read into bytes
    data = io.BytesIO(b"".join(chunks))
    return data


def _download(
    url: str,
    verbose: bool = False,
    retries: int = 5,
    wait_time: int = 20,
) -> io.BytesIO:
    m = f"Downloading {url}"
    _log(m, level="info", verbose=verbose)
    data = None
    for attempt in range(1, retries + 1):
        try:
            data = _download_chunks(url, verbose=verbose)
            break
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            if status_code == 429 and attempt < retries:
                _log(
                    f"429 Too Many Requests for {url}. Retrying in {wait_time}s (attempt {attempt + 1}/{retries})",
                    level="warn",
                    verbose=verbose,
                )
                time.sleep(wait_time)
                continue
            raise  # Not a 429 or no retries left: re-raise
    m = "Download finished"
    _log(m, level="info", verbose=verbose)
    return data


def _bytes_to_pandas(data: io.BytesIO, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(data, **kwargs)
    return df
