import logging
import tensorflow as tf
from cunet.callbacks import earlystopping, checkpoint
from cunet.config import config as config_train
from cunet.models.config import config as config_model
from cunet.models.cunet_model import cunet_model
from cunet.models.unet_model import unet_model
import os

logger = tf.get_logger()
logger.setLevel(logging.INFO)


def main():
    config_model.parse_args()
    name = config_model.MODE
    if config_model.MODE == 'standard':
        model = unet_model()
        name = "_".join([name, config_train.TARGET])
    if config_model.MODE == 'conditioned':
        model = cunet_model()
        name = "_".join([
            name, config_model.CONTROL_TYPE, config_model.FILM_TYPE,
            config_train.TRAIN_TYPE, str(config_train.CONDITION_MIX)
        ])
    # train_ds =
    # val_ds =
    model.fit_generator(
        generator=train_ds,
        steps_per_epoch=config_train.N_BATCH,
        epochs=config_train.N_EPOCH,
        validation_data=val_ds,
        workers=1,
        use_multiprocessing=False,
        callbacks=[earlystopping, checkpoint])
    model.save(os.path.join(config_train.PATH_OUTPUT, name))
    return


if __name__ == '__main__':
    main()
