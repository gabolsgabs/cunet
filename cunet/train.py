import logging
import tensorflow as tf
from cunet.callbacks import earlystopping, checkpoint
from cunet.config import config as config
from cunet.models.cunet_model import cunet_model
from cunet.models.unet_model import unet_model
from cunet.data_loader import dataset_generator
import os

logger = tf.get_logger()
logger.setLevel(logging.INFO)


def main():
    config.parse_args()
    name = config.MODE
    if config.MODE == 'standard':
        model = unet_model()
        name = "_".join([name, config.TARGET])
    if config.MODE == 'conditioned':
        model = cunet_model()
        name = "_".join([
            name, config.CONTROL_TYPE, config.FILM_TYPE,
            config.TRAIN_TYPE, str(config.CONDITION_MIX)
        ])
    train_ds = dataset_generator()
    val_ds = dataset_generator()
    model.fit_generator(
        generator=train_ds,
        steps_per_epoch=config.N_BATCH,
        epochs=config.N_EPOCH,
        validation_data=val_ds,
        workers=1,
        use_multiprocessing=False,
        callbacks=[earlystopping, checkpoint])
    model.save(os.path.join(config.PATH_OUTPUT, name))
    return


if __name__ == '__main__':
    main()
