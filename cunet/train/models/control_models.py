from tensorflow.keras.layers import (
    Input, Conv1D, Dense, BatchNormalization, Dropout
)
import tensorflow as tf
from cunet.train.config import config


def dense_block(
    x, n_neurons, input_dim, initializer, activation='relu'
):
    for i, (n, d) in enumerate(zip(n_neurons, input_dim)):
        extra = i != 0
        x = Dense(n, input_dim=d, activation=activation,
                  kernel_initializer=initializer)(x)
        if extra:
            x = Dropout(0.5)(x)
            x = BatchNormalization(momentum=0.9, scale=True)(x)
    return x


def dense_control(n_conditions, n_neurons):
    """
    For simple dense control:
        - n_conditions = 6
        - n_neurons = [16, 64, 256]
    For complex dense control:
        - n_conditions = 1008
        - n_neurons = [16, 128, 1024]

    """
    input_conditions = Input(shape=(1, config.Z_DIM))
    input_dim = [config.Z_DIM] + n_neurons[:-1]
    initializer = tf.random_normal_initializer(stddev=0.02)
    dense = dense_block(input_conditions, n_neurons, input_dim, initializer)
    gammas = Dense(
        n_conditions, input_dim=n_neurons[-1], activation=config.ACT_G,
        kernel_initializer=initializer
    )(dense)
    betas = Dense(
        n_conditions, input_dim=n_neurons[-1], activation=config.ACT_B,
        kernel_initializer=initializer
    )(dense)
    # both = Add()([gammas, betas])
    return input_conditions, gammas, betas


def cnn_block(
    x, n_filters, kernel_size, padding, initializer, activation='relu'
):
    for i, (f, p) in enumerate(zip(n_filters, padding)):
        extra = i != 0
        x = Conv1D(f, kernel_size, padding=p, activation=activation,
                   kernel_initializer=initializer)(x)
        if extra:
            x = Dropout(0.5)(x)
            x = BatchNormalization(momentum=0.9, scale=True)(x)
    return x


def cnn_control(n_conditions, n_filters):
    """
    For simple dense control:
        - n_conditions = 6
        - n_filters = [16, 32, 128]
    For complex dense control:
        - n_conditions = 1008
        - n_filters = [16, 32, 64]

    """
    input_conditions = Input(shape=(config.Z_DIM, 1))
    initializer = tf.random_normal_initializer(stddev=0.02)
    cnn = cnn_block(
        input_conditions, n_filters, config.Z_DIM, config.PADDING, initializer
    )
    gammas = Dense(
        n_conditions, input_dim=n_filters[-1], activation=config.ACT_G,
        kernel_initializer=initializer
    )(cnn)
    betas = Dense(
        n_conditions, input_dim=n_filters[-1], activation=config.ACT_B,
        kernel_initializer=initializer
    )(cnn)
    # both = Add()([gammas, betas])
    return input_conditions, gammas, betas
