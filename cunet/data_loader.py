import copy
from glob import glob
import numpy as np
import os
import tensorflow as tf
from cunet.config import config
from cunet.preprocess.config import config as config_pre
import random
import logging


logger = logging.getLogger('tensorflow')


def check_shape(data):
    n = data.shape[0]
    if n % 2 != 0:
        n = data.shape[0] - 1
    return np.expand_dims(data[:n, :], axis=2)


def get_name(txt):
    return os.path.basename(os.path.normpath(txt)).replace('.npz', '')


def complex_max(d):
    return d[np.unravel_index(np.argmax(np.abs(d), axis=None), d.shape)]


def complex_min(d):
    return d[np.unravel_index(np.argmin(np.abs(d), axis=None), d.shape)]


def normlize_complex(data):
    return np.divide((data - complex_min(data)),
                     (complex_max(data) - complex_min(data)))


def progressive(data, conditions, dx):
    output = copy.deepcopy(data)
    if (
        config.PROGRESSIVE and np.abs(complex_max(data)) > 0
        and np.sum((np.random.randint(4, size=1))) == 0   # 25% of doing it
    ):
        p = random.uniform(0, 1)
        conditions[dx] = conditions[dx]*p
    output = output[:, :, dx]*conditions[dx]
    return output, conditions


def load_files(files, val_files, val_set=False):
    data = {}
    val_files = [i.decode("utf-8") for i in val_files]
    if not val_set:
        files = [i for i in files if get_name(i) not in val_files]
    else:
        files = [i for i in files if get_name(i) in val_files]
    sources = []
    for i in files:
        logger.info('Loading the file %s' % i)
        data_tmp = np.load(i, allow_pickle=True)
        if config.MODE == 'standard':
            data[get_name(i)] = np.empty(
                [*data_tmp['mix'].shape, 2], dtype=np.complex64
            )
            data[get_name(i)][:, :, 0] = normlize_complex(
                data_tmp[config.SOURCE])
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
    return data, [get_name(i) for i in files]


def yield_data(indexes, data, files):
    conditions = np.zeros(1).astype(np.float32)
    n_frames = config.INPUT_SHAPE[1]
    for i in indexes['indexes']:
        if i[0] in files:
            if len(i) > 2:
                conditions = i[2]
            yield {'data': data[i[0]][:, i[1]:i[1]+n_frames, :],
                   'conditions': conditions}


def load_indexes_file(val_files, val_set=False):
    data, files = load_files(
        glob(os.path.join(config_pre.PATH_SPEC, '*.npz')), val_files, val_set
    )
    logger.info('Data loaded!')
    print('Data loaded!')
    if not val_set:
        indexes = np.load(os.path.join(config.PATH_BASE, config.INDEXES_TRAIN),
                          allow_pickle=True)
        while True:
            return yield_data(indexes, data, files)
    else:
        # Indexes val has no overlapp in the data points
        indexes = np.load(os.path.join(config.PATH_BASE, config.INDEXES_VAL),
                          allow_pickle=True)
        return yield_data(indexes, data, files)


@tf.function(autograph=False)
def prepare_data(data):
    def py_prepare_data(target_complex, conditions):
        target_complex = target_complex.numpy()
        conditions = conditions.numpy()
        if config.MODE == 'standard':
            target = np.abs(target_complex[:, :, 0])
        if config.MODE == 'conditioned':
            i = np.nonzero(conditions)[0]
            target = np.zeros(target_complex.shape[:2]).astype(np.complex64)
            if len(i) > 0:
                # simple conditions
                if len(i) == 1:
                    target, conditions = progressive(
                        target_complex, conditions, i[0]
                    )
                # complex conditions
                if len(i) > 1:
                    for dx in i:
                        target_tmp, conditions = progressive(
                            target_complex, conditions, dx
                        )
                        target = np.sum([target, target_tmp], axis=0)
            target = np.abs(target)*np.max(conditions)
        mixture = np.abs(target_complex[:, :, -1])
        return check_shape(mixture), check_shape(target), conditions
    mixture, target, conditions = tf.py_function(
        py_prepare_data, [data['data'], data['conditions']],
        (tf.float32, tf.float32, tf.float32)
    )
    return {'mix': mixture, 'target': target, 'conditions': conditions}


def convert_to_estimator_input(d):
    # just the mixture standar mode
    inputs = tf.ensure_shape(d["mix"], config.INPUT_SHAPE)
    if config.MODE == 'conditioned':
        if config.CONTROL_TYPE == 'dense':
            c_shape = (1, config.Z_DIM)
        if config.CONTROL_TYPE == 'cnn':
            c_shape = (config.Z_DIM, 1)
        cond = tf.ensure_shape(tf.reshape(d['conditions'], c_shape), c_shape)
        # mixture + condition vector z
        inputs = (inputs, cond)
        # target -> isolate instrument
    return (inputs, tf.ensure_shape(d["target"], config.INPUT_SHAPE))


def dataset_generator(val_files, val_set=False):
    ds = tf.data.Dataset.from_generator(
        load_indexes_file,
        {'data': tf.complex64, 'conditions': tf.float32},
        args=[val_files, val_set]
    ).map(
        prepare_data, num_parallel_calls=config.NUM_THREADS
    ).map(
        convert_to_estimator_input, num_parallel_calls=config.NUM_THREADS
    )
    ds = ds.batch(config.BATCH_SIZE, drop_remainder=True)
    if not val_set:
        ds = ds.repeat()
    return ds
