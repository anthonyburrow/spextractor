import logging
import os.path


_log_dir = './log'


def setup_log(filename, verbose=False, *args, **kwargs):
    if not os.path.exists(_log_dir):
        os.makedirs(_log_dir)

    if filename is None:
        index = 0
        while True:
            filename = f'{_log_dir}/sn{index}.log'
            if os.path.isfile(filename):
                index += 1
                continue
            break

    # Root logger
    logging.basicConfig(format='')
    logger = logging.getLogger('SPEXTRACTOR')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter('%(levelname)s: %(message)s')

    logger.handlers = []

    # File handler
    filename = f'{_log_dir}/{os.path.basename(filename)}'
    fh = logging.FileHandler(filename, mode='w')
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # Console handler
    if not verbose:
        return logger

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    return logger
