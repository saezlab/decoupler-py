"""
General purpose utility functions.
"""

import logging
import traceback
import warnings

__all__ = [
    "log_traceback",
]


def log_traceback(msg: str) -> None:
    """
    Print msg and the last traceback to the console and log.
    """

    exc = traceback.format_exc()
    logging.getLogger().warning(exc)
    logging.warning(msg)
    warnings.warn(msg)
