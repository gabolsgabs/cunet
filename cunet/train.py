import logging
import tensorflow as tf
from cunet.config import config
from cunet.models.cunet_model import cunet_model
from cunet.models.unet_model import unet_model

logger = tf.get_logger()
logger.setLevel(logging.INFO)


def main():
    config.parse_args()
    name = config.MODE
    if config.MODE == 'standard':
        model = unet_model()
    if config.MODE == 'conditioned':
        model = cunet_model()
        name = "_".join([name, config.CONTROL_TYPE, config.FILM_TYPE])
    return


if __name__ == '__main__':
    main()
