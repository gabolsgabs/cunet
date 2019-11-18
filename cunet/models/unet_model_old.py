import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, multiply,
    BatchNormalization, LeakyReLU, Dropout, Concatenate
)
from tensorflow.keras.optimizers import Adam


def get_activation(name):
    if name == 'leaky_relu':
        return LeakyReLU(alpha=0.2)
    return tf.keras.activations.get(name)


def u_net_conv_block(x, nb_filter, initializer, activation='leaky_relu'):
    x = Conv2D(nb_filter, (5, 5),  padding='same', strides=(2, 2),
               kernel_initializer=initializer)(x)
    x = BatchNormalization(momentum=0.9, scale=True)(x)
    x = get_activation(activation)(x)
    return x


def u_net_deconv_block(x_decod, x_encod, nb_filter, initializer, activation,
                       dropout, skip):
    x = x_encod
    if skip:
        x = Concatenate(axis=3)([x_decod, x])
    x = Conv2DTranspose(
        nb_filter, 5, padding='same', strides=2,
        kernel_initializer=initializer)(x)
    x = BatchNormalization(momentum=0.9, scale=True)(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = get_activation(activation)(x)
    return x


def unet_model(input_shape=(512, 128, 1), filters_layer_1=16, n_layers=6,
               lr=0.001, act_last='sigmoid', **kargs):
    n_freq, n_frame, _ = input_shape
    # axis should be fr, time -> right not it's time freqs
    inputs = Input(shape=input_shape)
    x = inputs
    encoder_layers = []
    initializer = tf.random_normal_initializer(stddev=0.02)
    # Encoder
    for i in range(n_layers):
        filters = filters_layer_1 * (2 ** i)
        x = u_net_conv_block(x, filters, initializer)
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
            filters = 1
            activation = act_last
        else:
            filters = encoder_layer.get_shape().as_list()[-1] // 2
            activation = 'relu'
        x = u_net_deconv_block(
            x, encoder_layer, filters, initializer, activation, dropout, skip)
    outputs = multiply([inputs, x])
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(lr=lr), loss='mean_absolute_error')
    return model
