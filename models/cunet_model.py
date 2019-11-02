import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, multiply,
    BatchNormalization, LeakyReLU, Dropout, Concatenate
)
from tensorflow.keras.optimizers import Adam
from models.FiLM_utils import (
    FiLM_simple_layer, FiLM_complex_layer, slice_tensor, slice_tensor_range
)
from models.control_models import dense_control, cnn_control


def get_activation(name):
    if name == 'leaky_relu':
        return LeakyReLU(alpha=0.2)
    return tf.keras.activations.get(name)


def u_net_conv_block(x, n_filters, initializer, gamma, beta,
                     activation='leaky_relu', film_type='simple'):
    x = Conv2D(n_filters, (5, 5),  padding='same', strides=(2, 2),
               kernel_initializer=initializer)(x)
    x = BatchNormalization(momentum=0.9, scale=True)(x)
    if film_type == 'simple':
        x = FiLM_simple_layer()([x, gamma, beta])
    if film_type == 'complex':
        x = FiLM_complex_layer()([x, gamma, beta])
    x = get_activation(activation)(x)
    return x


def u_net_deconv_block(x_decod, x_encod, n_filters, initializer, activation,
                       dropout, skip):
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


def cunet_model(
    input_shape=(512, 128, 1), filters_layer_1=16, n_layers=6, lr=0.001,
    act_last='sigmoid', control_type='dense', film_type='simple', **kargs
):
    n_freq, n_frame, _ = input_shape
    # axis should be fr, time -> right not it's time freqs
    inputs = Input(shape=input_shape)
    x = inputs
    encoder_layers = []
    initializer = tf.random_normal_initializer(stddev=0.02)

    if control_type == 'dense' and film_type == 'simple':
        input_conditions, gammas, betas = dense_control(
            n_conditions=6, n_neurons=[16, 64, 256])
    if control_type == 'cnn' and film_type == 'simple':
        input_conditions, gammas, betas = cnn_control(
            n_conditions=6, n_filters=[16, 32, 64])
    if control_type == 'dense' and film_type == 'complex':
        input_conditions, gammas, betas = dense_control(
            n_conditions=1008, n_neurons=[16, 256, 1024])
    if control_type == 'cnn' and film_type == 'complex':
        input_conditions, gammas, betas = cnn_control(
            n_conditions=1008, n_filters=[32, 64, 252])

    # Encoder
    complex_index = 0
    for i in range(n_layers):
        filters = filters_layer_1 * (2 ** i)
        if film_type == 'simple':
            gamma, beta = slice_tensor(i)(gammas), slice_tensor(i)(betas)
        if film_type == 'complex':
            init, end = complex_index, complex_index+filters
            gamma = slice_tensor_range(init, end)(gammas)
            beta = slice_tensor_range(init, end)(betas)
            complex_index += filters
        x = u_net_conv_block(
            x, filters, initializer, gamma, beta, film_type=film_type)
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
            filters = 1
            activation = act_last
        else:
            filters = encoder_layer.get_shape().as_list()[-1] // 2
            activation = 'relu'
        x = u_net_deconv_block(
            x, encoder_layer, filters, initializer, activation, dropout, skip)
    outputs = multiply([inputs, x])
    model = Model(inputs=[inputs, input_conditions], outputs=outputs)
    model.compile(optimizer=Adam(lr=lr), loss='mean_absolute_error')
    return model
