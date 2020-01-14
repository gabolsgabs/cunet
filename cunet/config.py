# -*- coding: utf-8 -*-
from effortless_config import Config, setting
import tensorflow as tf


class config(Config):

    groups = ['simple_dense', 'complex_dense', 'simple_cnn', 'complex_cnn']
    # General

    MODE = setting(
        'standard',
        simple_dense='conditioned', complex_dense='conditioned',
        simple_cnn='conditioned', complex_cnn='conditioned'
    )

    SOURCE = setting(
        default='vocals',   # only for standard version
    )
    NAME = 'test'
    ADD_TIME = False

    # GENERATOR
    PATH_BASE = '/data2/anasynth_nonbp/meseguerbrocal/source_separation/musdb18/'
    # INDEXES_TRAIN = 'train/indexes/indexes_standard_1_4.npz'
    INDEXES_TRAIN = 'train/indexes/indexes_conditioned_1_4_1_True_True_1.0.npz'
    # INDEXES_VAL = 'train/indexes/indexes_standard_128_4.npz'
    INDEXES_VAL = 'train/indexes/indexes_conditioned_128_4_1_True_True_1.0.npz'
    NUM_THREADS = tf.data.experimental.AUTOTUNE   #
    N_PREFETCH = 2048  # tf.data.experimental.AUTOTUNE

    # checkpoints
    EARLY_STOPPING_MIN_DELTA = 0.0
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_PLATEAU_PATIENCE = 5

    # training
    BATCH_SIZE = 64
    N_BATCH = 1024
    N_EPOCH = 1000
    PROGRESSIVE = True

    # unet paramters
    INPUT_SHAPE = [512, 128, 1]  # freq = 512, time = 128
    FILTERS_LAYER_1 = 16
    N_LAYERS = 6
    LR = 1e-4
    ACTIVATION_ENCODER = 'leaky_relu'
    ACTIVATION_DECODER = 'relu'
    ACT_LAST = 'sigmoid'
    LOSS = 'mean_absolute_error'

    # -------------------------------

    # control parameters
    CONTROL_TYPE = setting(
        'dense', simple_dense='dense', complex_dense='dense',
        simple_cnn='cnn', complex_cnn='cnn'
    )
    FILM_TYPE = setting(
        'simple', simple_dense='simple', complex_dense='complex',
        simple_cnn='simple', complex_cnn='complex'
    )
    Z_DIM = 4       # for musdb -> 4 instruments: vocals, drums, bass, rest
    ACT_G = 'linear'
    ACT_B = 'linear'
    N_CONDITIONS = setting(
        6, simple_dense=6, complex_dense=1008,
        simple_cnn=6, complex_cnn=1008
    )

    # cnn control
    N_FILTERS = setting(
        [16, 32, 64], simple_cnn=[16, 32, 64], complex_cnn=[32, 64, 256]
    )
    PADDING = ['same', 'same', 'valid']
    # Dense control
    N_NEURONS = setting(
        [16, 64, 256], simple_dense=[16, 64, 256],
        complex_dense=[16, 256, 1024]
    )
