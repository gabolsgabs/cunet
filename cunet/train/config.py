# -*- coding: utf-8 -*-
from effortless_config import Config, setting
import tensorflow as tf
import os


class config(Config):

    groups = ['standard', 'simple_dense', 'complex_dense', 'simple_cnn',
              'complex_cnn']
    # General

    MODE = setting(default='conditioned', standard='standard')

    NAME = 'with_new_norm_and_aug'
    ADD_TIME = False    # add the time and date in the name
    TARGET = 'vocals'   # only for standard version

    # GENERATOR
    PATH_BASE = '/net/guzheng/data2/anasynth_nonbp/meseguerbrocal/source_separation/musdb18/'
    # default = conditioned
    INDEXES_TRAIN = setting(
        default=os.path.join(
            PATH_BASE, 'train/indexes/indexes_conditioned_1_4_1_False_False_1.0.npz'),
        standard=os.path.join(
            PATH_BASE, 'train/indexes/indexes_standard_1_4.npz')
    )
    INDEXES_VAL = setting(
        default=os.path.join(
            PATH_BASE, 'train/indexes/indexes_conditioned_128_4_1_False_False_1.0.npz'),
        standard=os.path.join(
            PATH_BASE, 'train/indexes/indexes_standard_128_4.npz')
    )

    NUM_THREADS = tf.data.experimental.AUTOTUNE   # 32
    N_PREFETCH = tf.data.experimental.AUTOTUNE  # 4096

    # checkpoints
    EARLY_STOPPING_MIN_DELTA = 1e-5
    EARLY_STOPPING_PATIENCE = 30
    REDUCE_PLATEAU_PATIENCE = 15

    # training
    BATCH_SIZE = 64
    N_BATCH = 2048
    N_EPOCH = 1000
    PROGRESSIVE = True
    AUG = True

    # unet paramters
    INPUT_SHAPE = [512, 128, 1]  # freq = 512, time = 128
    FILTERS_LAYER_1 = 16
    N_LAYERS = 6
    LR = 1e-3
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
