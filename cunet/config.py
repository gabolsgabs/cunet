# -*- coding: utf-8 -*-
from effortless_config import Config, setting


class config(Config):
    groups = ['simple_dense', 'complex_dense', 'simple_cnn', 'complex_cnn']

    MODE = setting(
        'standard',
        simple_dense='conditioned', complex_dense='conditioned',
        simple_cnn='conditioned', complex_cnn='conditioned'
    )

    # unet paramters
    INPUT_SHAPE = [512, 128, 1]
    FILTERS_LAYER_1 = 16
    N_LAYERS = 6
    LR = 0.001
    ACTIVATION_ENCODER = 'leaky_relu'
    ACTIVATION_DECODER = 'relu'
    ACT_LAST = 'sigmoid'
    LOSS = 'mean_absolute_error'

    # control parameters
    CONTROL_TYPE = setting(
        'dense', simple_dense='dense', complex_dense='dense',
        simple_cnn='cnn', complex_cnn='cnn'
    )
    FILM_TYPE = setting(
        'simple', simple_dense='simple', complex_dense='complex',
        simple_cnn='simple', complex_cnn='complex'
    )
    Z_DIM = 4
    ACT_G = 'linear'
    ACT_B = 'linear'
    N_CONDITIONS = setting(
        6, simple_dense=6, complex_dense=1008,
        simple_cnn=6, complex_cnn=1008
    )

    # CNN control
    N_FILTERS = setting(
        [16, 32, 64], simple_cnn=[16, 32, 64], complex_cnn=[32, 64, 256]
    )
    PADDING = ['same', 'same', 'valid']
    # Dense control
    N_NEURONS = setting(
        [16, 64, 256], simple_dense=[16, 64, 256],
        complex_dense=[16, 256, 1024]
    )
