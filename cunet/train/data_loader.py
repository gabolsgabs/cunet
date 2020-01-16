import copy
import numpy as np
import os
import tensorflow as tf
from cunet.train.config import config
from cunet.train.load_data_offline import DATA
from cunet.train.others.val_files import VAL_FILES
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
        config.PROGRESSIVE and np.max(np.abs(data)) > 0
        and random.sample(range(0, 4), 1)[0] == 0   # 25% of doing it
    ):
        p = random.uniform(0, 1)
        conditions[dx] = conditions[dx]*p
    output = output[:, :, dx]*conditions[dx]
    return output, conditions


def yield_data(indexes, files):
    conditions = np.zeros(1).astype(np.float32)
    n_frames = config.INPUT_SHAPE[1]
    for i in indexes:
        if i[0] in files:
            if len(i) > 2:
                conditions = i[2]
            yield {'data': DATA[i[0]][:, i[1]:i[1]+n_frames, :],
                   'conditions': conditions}


def load_indexes_file(val_set=False):
    if not val_set:
        indexes = np.load(config.INDEXES_TRAIN, allow_pickle=True)['indexes']
        r = list(range(len(indexes)))
        random.shuffle(r)
        indexes = indexes[r]
        files = [i for i in DATA.keys() if i is not VAL_FILES]
    else:
        # Indexes val has no overlapp in the data points
        indexes = np.load(config.INDEXES_VAL, allow_pickle=True)['indexes']
        files = VAL_FILES
    return yield_data(indexes, files)


@tf.function(autograph=False)
def prepare_data(data):
    def py_prepare_data(target_complex, conditions):
        target_complex = target_complex.numpy()
        conditions = conditions.numpy()
        if config.MODE == 'standard':
            # the instrument is already selected and normalized in load_files
            target = np.abs(target_complex[:, :, 0])    # thus target in 0
        if config.MODE == 'conditioned':
            i = np.nonzero(conditions)[0]
            target = np.zeros(target_complex.shape[:2]).astype(np.complex64)
            if len(i) > 0:
                # simple conditions
                if len(i) == 1:
                    target, conditions = progressive(
                        target_complex, conditions, i[0])
                # complex conditions
                if len(i) > 1:
                    for dx in i:
                        target_tmp, conditions = progressive(
                            target_complex, conditions, dx)
                        target = np.sum([target, target_tmp], axis=0)
            if np.max(conditions) != 1:
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
    outputs = tf.ensure_shape(d["target"], config.INPUT_SHAPE)
    return (inputs, outputs)


def dataset_generator(val_set=False):
    ds = tf.data.Dataset.from_generator(
        load_indexes_file,
        {'data': tf.complex64, 'conditions': tf.float32},
        args=[val_set]
    ).map(
        prepare_data, num_parallel_calls=config.NUM_THREADS
    ).map(
        convert_to_estimator_input, num_parallel_calls=config.NUM_THREADS
    ).batch(
        config.BATCH_SIZE, drop_remainder=True
    ).prefetch(
        buffer_size=config.N_PREFETCH
    )
    if not val_set:
        ds = ds.repeat()
    return ds
