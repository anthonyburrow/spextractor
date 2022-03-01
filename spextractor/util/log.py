import logging
import os.path


def setup_log(filename, verbose=True):
    if not os.path.exists('./log/'):
        os.makedirs('./log/')

    if filename is None:
        index = 0
        while True:
            filename = './log/sn%i.log' % index
            if os.path.isfile(filename):
                index += 1
                continue
            break

    # Root logger
    logging.basicConfig(format='')
    root = logging.getLogger('SPEXTRACTOR')
    root.setLevel(logging.INFO)
    root.propagate = False

    formatter = logging.Formatter('%(levelname)s: %(message)s')

    # File handler
    filename = './log/%s' % os.path.basename(filename)
    fh = logging.FileHandler(filename, mode='w')
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    root.addHandler(fh)

    # Console handler
    if verbose:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)
        root.addHandler(ch)

    return root
