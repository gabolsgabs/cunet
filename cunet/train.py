import logging
import tensorflow as tf
from cunet.utilities import (
    make_earlystopping, make_reduce_lr, make_tensorboard, make_checkpoint,
    make_name, save_dir
)
from cunet.config import config as config
from cunet.models.cunet_model import cunet_model
from cunet.models.unet_model import unet_model
from cunet.data_loader import dataset_generator
from cunet.val_files import val_files
import os


logger = tf.get_logger()
logger.setLevel(logging.INFO)


def main():
    config.parse_args()
    name = make_name()

    logger.info('Starting the computation')

    logger.info('Running training with config %s' % str(config))
    logger.info('Getting the model')
    if config.MODE == 'standard':
        model = unet_model()
    if config.MODE == 'conditioned':
        model = cunet_model()

    logger.info('Preparing the genrators')
    ds_train = dataset_generator(val_files)
    ds_val = dataset_generator(val_files, val_set=True)

    logger.info('Starting training for %s' % name)
    model.fit(
        ds_train,
        validation_data=ds_val,
        steps_per_epoch=config.N_BATCH,
        epochs=config.N_EPOCH,
        callbacks=[
            make_earlystopping(),
            make_reduce_lr(),
            make_tensorboard(name),
            make_checkpoint(name)
        ])

    logger.info('Saving model %' % name)
    model.save(os.path.join((save_dir('models'), name+'.h5')))
    logger.info('Done!')

    return


if __name__ == '__main__':
    main()
