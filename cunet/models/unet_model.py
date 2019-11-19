import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, multiply,
    BatchNormalization, LeakyReLU, Dropout, Concatenate
)
from tensorflow.keras.optimizers import Adam
from cunet.models.config import config


def get_activation(name):
    if name == 'leaky_relu':
        return LeakyReLU(alpha=0.2)
    return tf.keras.activations.get(name)


def u_net_conv_block(x, n_filters, initializer, activation):
    x = Conv2D(n_filters, (5, 5),  padding='same', strides=(2, 2),
               kernel_initializer=initializer)(x)
    x = BatchNormalization(momentum=0.9, scale=True)(x)
    x = get_activation(activation)(x)
    return x


def u_net_deconv_block(
    x_decod, x_encod, n_filters, initializer, activation, dropout, skip
):
    x = x_encod
    if skip:
        x = Concatenate(axis=3)([x_decod, x])
    x = Conv2DTranspose(
        n_filters, 5, padding='same', strides=2,
        kernel_initializer=initializer)(x)
    x = BatchNormalization(momentum=0.9, scale=True)(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = get_activation(activation)(x)
    return x


def unet_model():
    # axis should be fr, time -> right not it's time freqs
    inputs = Input(shape=config.INPUT_SHAPE)
    n_layers = config.N_LAYERS
    x = inputs
    encoder_layers = []
    initializer = tf.random_normal_initializer(stddev=0.02)
    # Encoder
    for i in range(n_layers):
        n_filters = config.FILTERS_LAYER_1 * (2 ** i)
        x = u_net_conv_block(
            x, n_filters, initializer, config.ACTIVATION_ENCODER
        )
        encoder_layers.append(x)
    # Decoder
    for i in range(n_layers):
        # parameters each decoder layer
        is_final_block = i == n_layers - 1  # the las layer is different
        # not dropout in the first block and the last two encoder blocks
        dropout = not (i == 0 or i == n_layers - 1 or i == n_layers - 2)
        # for getting the number of filters
        encoder_layer = encoder_layers[n_layers - i - 1]
        skip = i > 0    # not skip in the first encoder block - the deepest
        if is_final_block:
            n_filters = 1
            activation = config.ACT_LAST
        else:
            n_filters = encoder_layer.get_shape().as_list()[-1] // 2
            activation = config.ACTIVATION_DECODER
        x = u_net_deconv_block(
            x, encoder_layer, n_filters, initializer, activation, dropout, skip
        )
    outputs = multiply([inputs, x])
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(lr=config.LR), loss=config.LOSS)
    return model
