import logging

logger = logging.getLogger("decoupler")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s | [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(_handler)


def _log(message: str, level: str = "info", verbose: bool = False) -> None:
    """
    Log a message with a specified logging level.

    Parameters
    ----------
    message
        The message to log.
    level
        The logging level.
    verbose
        Whether to emit the log.
    """
    level = level.lower()
    if verbose:
        if level == "warn":
            logger.warning(message)
        elif level == "info":
            logger.info(message)
