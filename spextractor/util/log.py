import logging
import os.path


_default_log_dir = './log'


def setup_log(filename, verbose=False, log_to_file=False, log_dir=None,
              *args, **kwargs):
    # Root logger
    logging.basicConfig(format='')
    logger = logging.getLogger('SPEXTRACTOR')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter('%(levelname)s: %(message)s')

    logger.handlers = []

    # File handler
    if log_to_file:
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
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

    # Console handler
    if verbose:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    return logger
