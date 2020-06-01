import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, multiply, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from cunet.train.models.FiLM_utils import (
    FiLM_simple_layer, FiLM_complex_layer, slice_tensor, slice_tensor_range
)
from cunet.train.models.control_models import dense_control, cnn_control
from cunet.train.models.unet_model import get_activation, u_net_deconv_block
from cunet.train.config import config


def u_net_conv_block(
    x, n_filters, initializer, gamma, beta, activation, film_type,
    kernel_size=(5, 5), strides=(2, 2), padding='same'
):
    x = Conv2D(n_filters, kernel_size=kernel_size,  padding=padding,
               strides=strides, kernel_initializer=initializer)(x)
    x = BatchNormalization(momentum=0.9, scale=True)(x)
    if film_type == 'simple':
        x = FiLM_simple_layer()([x, gamma, beta])
    if film_type == 'complex':
        x = FiLM_complex_layer()([x, gamma, beta])
    x = get_activation(activation)(x)
    return x


def cunet_model():
    inputs = Input(shape=config.INPUT_SHAPE)
    n_layers = config.N_LAYERS
    x = inputs
    encoder_layers = []
    initializer = tf.random_normal_initializer(stddev=0.02)

    if config.CONTROL_TYPE == 'dense':
        input_conditions, gammas, betas = dense_control(
            n_conditions=config.N_CONDITIONS, n_neurons=config.N_NEURONS)
    if config.CONTROL_TYPE == 'cnn':
        input_conditions, gammas, betas = cnn_control(
            n_conditions=config.N_CONDITIONS, n_filters=config.N_FILTERS)
    # Encoder
    complex_index = 0
    for i in range(n_layers):
        n_filters = config.FILTERS_LAYER_1 * (2 ** i)
        if config.FILM_TYPE == 'simple':
            gamma, beta = slice_tensor(i)(gammas), slice_tensor(i)(betas)
        if config.FILM_TYPE == 'complex':
            init, end = complex_index, complex_index+n_filters
            gamma = slice_tensor_range(init, end)(gammas)
            beta = slice_tensor_range(init, end)(betas)
            complex_index += n_filters
        x = u_net_conv_block(
            x, n_filters, initializer, gamma, beta,
            activation=config.ACTIVATION_ENCODER, film_type=config.FILM_TYPE
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
        skip = i > 0    # not skip in the first encoder block
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
    model = Model(inputs=[inputs, input_conditions], outputs=outputs)
    model.compile(
        optimizer=Adam(lr=config.LR, beta_1=0.5), loss=config.LOSS)
        # experimental_run_tf_function=False)
    return model
