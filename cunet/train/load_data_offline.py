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


def normlize_complex(data):
    return np.divide((data - complex_min(data)),
                     (complex_max(data) - complex_min(data)))


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
            data[get_name(i)][:, :, 0] = normlize_complex(
                data_tmp[config.TARGET])
            data[get_name(i)][:, :, 1] = normlize_complex(
                data_tmp['mix'])
        if config.MODE == 'conditioned':
            if len(sources) == 0:
                sources = copy.deepcopy(data_tmp.files)
                sources.remove('config')
                # to be sure that the mix is the last element
                sources.insert(len(sources), sources.pop(sources.index('mix')))
            data[get_name(i)] = np.empty(
                [*data_tmp['mix'].shape, len(sources)], dtype=np.complex64
            )
            for j, value in enumerate(sources):
                data[get_name(i)][:, :, j] = normlize_complex(data_tmp[value])
    if config.MODE == 'conditioned':
        logger.info('Source order %s' % sources)
    return data


def get_data():
    return load_data(
        glob(os.path.join(config_pre.PATH_SPEC, '*.npz'))
    )
