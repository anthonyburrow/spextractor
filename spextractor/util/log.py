import logging
from logging import Logger
import os.path


_default_log_dir = './log'

_log_format = logging.Formatter('%(levelname)s: %(message)s')


def add_file_handler(
    logger: Logger,
    filename: str | None = None,
    log_dir: str | None = None,
    *args,
    **kwargs,
):
    if log_dir is None:
        log_dir = _default_log_dir

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if filename is None:
        index = 0
        while True:
            filename = f'{log_dir}/sn{index}.log'
            if os.path.isfile(filename):
                index += 1
                continue
            break

    filename = f'{log_dir}/{os.path.basename(filename)}'
    fh = logging.FileHandler(filename, mode='w')

    fh.setFormatter(_log_format)
    fh.setLevel(logging.INFO)

    logger.addHandler(fh)

    return logger


def add_console_handler(logger: Logger, *args, **kwargs) -> Logger:
    ch = logging.StreamHandler()

    ch.setFormatter(_log_format)
    ch.setLevel(logging.INFO)

    logger.addHandler(ch)

    return logger


def setup_log(
    verbose: bool = False, log_to_file: bool = False, *args, **kwargs
) -> Logger:
    # Root logger
    logging.basicConfig(format='')
    logger = logging.getLogger('SPEXTRACTOR')

    logger.setLevel(logging.INFO)
    logger.propagate = False

    logger.handlers = []

    if log_to_file:
        logger = add_file_handler(logger, *args, **kwargs)

    if verbose:
        logger = add_console_handler(logger, *args, **kwargs)

    return logger
