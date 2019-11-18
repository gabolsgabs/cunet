from tensorflow.keras.layers import (
    Input, Conv1D, Dense, BatchNormalization, Dropout
)


def dense_block(
    x, n_neurons, input_dim, initializer='he_normal', activation='relu'
):
    for i, (n, d) in enumerate(zip(n_neurons, input_dim)):
        extra = i != 0
        x = Dense(n, input_dim=d, activation=activation,
                  kernel_initializer=initializer)(x)
        if extra:
            x = Dropout(0.5)(x)
            x = BatchNormalization(momentum=0.9, scale=True)(x)
    return x


def dense_control(
    n_conditions, n_neurons, z_dim=4, act_g='linear', act_b='linear',
    initializer='he_normal',
):
    """
    For simple dense control:
        - n_conditions = 6
        - n_neurons = [16, 64, 256]
    For complex dense control:
        - n_conditions = 1008
        - n_neurons = [16, 128, 1024]
    """
    input_conditions = Input(shape=(1, z_dim))
    input_dim = [z_dim] + n_neurons[:-1]
    dense = dense_block(input_conditions, n_neurons, input_dim)
    gammas = Dense(n_conditions, input_dim=n_neurons[-1], activation=act_g,
                   kernel_initializer=initializer)(dense)
    betas = Dense(n_conditions, input_dim=n_neurons[-1], activation=act_b,
                  kernel_initializer=initializer)(dense)
    # both = Add()([gammas, betas])
    return input_conditions, gammas, betas


def cnn_block(
    x, n_filters, kernel_size, padding, initializer='he_normal',
    activation='relu'
):
    for i, (f, p) in enumerate(zip(n_filters, padding)):
        extra = i != 0
        x = Conv1D(f, kernel_size, padding=p, activation=activation,
                   kernel_initializer=initializer)(x)
        if extra:
            x = Dropout(0.5)(x)
            x = BatchNormalization(momentum=0.9, scale=True)(x)
    return x


def cnn_control(
    n_conditions, n_filters, padding=['same', 'same', 'valid'], z_dim=4,
    act_g='linear', act_b='linear', initializer='he_normal'
):
    """
    For simple dense control:
        - n_conditions = 6
        - n_filters = [16, 32, 128]
    For complex dense control:
        - n_conditions = 1008
        - n_filters = [16, 32, 64]
    """
    input_conditions = Input(shape=(z_dim, 1))
    cnn = cnn_block(input_conditions, n_filters, z_dim, padding)
    gammas = Dense(n_conditions, input_dim=n_filters[-1], activation=act_g,
                   kernel_initializer=initializer)(cnn)
    betas = Dense(n_conditions, input_dim=n_filters[-1], activation=act_b,
                  kernel_initializer=initializer)(cnn)
    # both = Add()([gammas, betas])
    return input_conditions, gammas, betas
