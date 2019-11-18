import logging
import tensorflow as tf
from cunet.config import config
from cunet.models.cunet_model import cunet_model
from cunet.models.unet_model import unet_model

from tensorflow.keras.utils import plot_model

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
    plot_model(
        model,
        to_file='/Users/meseguerbrocal/Documents/PhD/code_bk_local/cunet/models/architectures/'+name+'.png',
        show_shapes=True,
        show_layer_names=True
    )
    return


if __name__ == '__main__':
    main()
