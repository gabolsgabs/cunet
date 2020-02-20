import copy
import numpy as np
import os
from cunet.train.config import config
import logging
from glob import glob
from cunet.preprocess.config import config as config_pre


logger = logging.getLogger('tensorflow')


def get_name(txt):
    return os.path.basename(os.path.normpath(txt)).replace('.npz', '')


def complex_max(d):
    return d[np.unravel_index(np.argmax(np.abs(d), axis=None), d.shape)]


def complex_min(d):
    return d[np.unravel_index(np.argmin(np.abs(d), axis=None), d.shape)]


def normlize_complex(data, c_max=1):
    if c_max != 1:
        factor = np.divide(complex_max(data), c_max)
    else:
        factor = 1
    # normalize between 0-1
    output = np.divide((data - complex_min(data)),
                       (complex_max(data) - complex_min(data)))
    return np.multiply(output, factor)  # scale to the original range


def get_max_complex(data, keys):
    # sometimes the max is not the mixture
    pos = np.argmax([np.abs(complex_max(data[i])) for i in keys])
    return np.array([complex_max(data[i]) for i in keys])[pos]


def load_data(files):
    """The data is loaded in memory just once for the generator to have direct
    access to it"""
    data = {}
    sources = []
    for i in files:
        logger.info('Loading the file %s' % i)
        data_tmp = np.load(i, allow_pickle=True)
        if config.MODE == 'standard':
            data[get_name(i)] = np.empty(
                [*data_tmp['mix'].shape, 2], dtype=np.complex64
            )
            c_max = get_max_complex(data_tmp, ['mix', config.TARGET])
            data[get_name(i)][:, :, 0] = normlize_complex(
                data_tmp[config.TARGET], c_max)
            data[get_name(i)][:, :, 1] = normlize_complex(
                data_tmp['mix'], c_max)
        if config.MODE == 'conditioned':
            if len(sources) == 0:
                sources = copy.deepcopy(data_tmp.files)
                sources.remove('config')
                # to be sure that the mix is the last element
                sources.insert(len(sources), sources.pop(sources.index('mix')))
            data[get_name(i)] = np.empty(
                [*data_tmp['mix'].shape, len(sources)], dtype=np.complex64
            )
            c_max = get_max_complex(data_tmp, sources)
            for j, value in enumerate(sources):
                data[get_name(i)][:, :, j] = normlize_complex(
                    data_tmp[value], c_max
                )
    if config.MODE == 'conditioned':
        logger.info('Source order %s' % sources)
    return data


def get_data():
    return load_data(
        glob(os.path.join(config_pre.PATH_SPEC, '*.npz'))
    )
