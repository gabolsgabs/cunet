import numpy as np
import copy
import itertools
import os
from cunet.preprocess.config import config
from cunet.train.config import config as config_train
from glob import glob
import logging


def condition2key(cond):
    key = np.array2string(cond).replace("[", "").replace("]", "")
    return ",".join([t for t in key.split(" ") if t != ''])


def get_conditions():
    logger = logging.getLogger('getting_indexes')
    logger.info('Computing the conditions')
    conditions_raw = [
        np.array(i).astype(np.float)
        for i in list(itertools.product([0, 1], repeat=4))
        if np.sum(i) <= config.CONDITION_MIX and np.sum(i) > 0
    ]
    conditions = []
    if config.ADD_ALL:      # add the all mix condition
        conditions_raw.append(np.ones(len(config.CONDITIONS)))
    keys = []
    in_between = np.arange(
        config.ADD_IN_BETWEEN, 1+config.ADD_IN_BETWEEN, config.ADD_IN_BETWEEN
    )
    for cond in conditions_raw:
        for index in np.nonzero(cond)[0]:
            # adding intermedia values to the conditions - in between idea
            for b in in_between:
                tmp = copy.deepcopy(cond)
                tmp[index] = tmp[index]*b
                key = condition2key(tmp)
                if key not in keys:     # avoiding duplicates
                    conditions.append(tmp.astype(np.float32))
                    keys.append(key)
    if config.ADD_ZERO:     # add the zero condition
        conditions.append(np.zeros(len(config.CONDITIONS)).astype(np.float32))
    logger.info('Done!')
    logger.info('Your conditions are %s', conditions)
    return conditions


def chunks(l, chunk_size):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), chunk_size):
        yield l[i:i + chunk_size]


def get_indexes(conditions, time_r):
    logger = logging.getLogger('getting_indexes')
    logger.info('Computing the indexes')
    indexes = []
    try:
        files = glob(os.path.join(config.PATH_SPEC, '*.npz'))
        for f in np.random.choice(files, len(files), replace=False):
            logger.info('Input points for track %s' % f)
            file_length = np.load(f)[config.INTRUMENTS[0]].shape[1]  # in frames
            s = []
            name = os.path.basename(os.path.normpath(f)).replace('.npz', '')
            for j in np.arange(0, file_length-time_r, config.STEP):
                if len(conditions) > 0:
                    for c in conditions:
                        s.append([name, j, c])
                else:
                    s.append([name, j])
            s = list(
                np.asarray(s, dtype=object)[np.random.permutation(len(s))]
            )
            indexes += s
        logger.info('Chunking the data points')
        # chunking the indexes before mixing -> create groups of CHUNK_SIZE
        indexes = list(chunks(indexes, config.CHUNK_SIZE))
        logger.info('Shuffling')
        # mixing these groups
        np.random.shuffle(indexes)
        # joining the groups in a single vector
        logger.info('Shuffling')
        indexes = list(itertools.chain.from_iterable(indexes))
    except Exception as error:
        logger.error(error)
    return indexes


def main():
    logging.basicConfig(
        filename=os.path.join(config.PATH_INDEXES, 'getting_indexes.log'),
        level=logging.INFO
    )
    logger = logging.getLogger('getting_indexes')
    logger.info('Starting the computation')
    freq_r, time_r = config_train.INPUT_SHAPE[:2]
    conditions = []
    name = "_".join([
        'indexes', config.MODE, str(config.STEP), str(config.CHUNK_SIZE)
    ])
    if config.MODE == 'conditioned':
        conditions = get_conditions()
        name = "_".join([
            name, str(config.CONDITION_MIX), str(config.ADD_ALL),
            str(config.ADD_ZERO), str(config.ADD_IN_BETWEEN)
        ])
    indexes = get_indexes(conditions, time_r)
    logger.info('Saving')
    np.savez(
        os.path.join(config.PATH_INDEXES, name),
        indexes=indexes, conditions=conditions, config=str(config)
    )
    logger.info('Done!')
    return


if __name__ == "__main__":
    config.parse_args()
    main()
