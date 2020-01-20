# -*- coding: utf-8 -*-
from effortless_config import Config, setting
import os


class config(Config):
    groups = ['standard', 'simple_dense', 'complex_dense', 'simple_cnn',
              'complex_cnn']

    GROUP = setting(
        default='simple_dense', simple_dense='simple_dense',
        complex_dense='complex_dense',
        simple_cnn='simple_cnn', complex_cnn='complex_cnn',
    )
    PATH_BASE = '/net/guzheng/data2/anasynth_nonbp/meseguerbrocal/source_separation/musdb18/models/'
    NAME = 'with_val_all_files'

    PATH_MODEL = setting(
        os.path.join(PATH_BASE, 'conditioned/simple_dense'),
        standard=os.path.join(PATH_BASE, 'standard'),
        simple_dense=os.path.join(PATH_BASE, 'conditioned/simple_dense'),
        complex_dense=os.path.join(PATH_BASE, 'conditioned/complex_dense'),
        simple_cnn=os.path.join(PATH_BASE, 'conditioned/simple_cnn'),
        complex_cnn=os.path.join(PATH_BASE, 'conditioned/complex_cnn')
    )
    PATH_AUDIO = '/net/guzheng/data2/anasynth_nonbp/meseguerbrocal/source_separation/musdb18/test/complex'
    TARGET = ['vocals', 'bass', 'drums', 'rest']  # ['vocals', 'bass', 'bass_vocals'] -> not ready yet for complex conditions
    INSTRUMENTS = ['bass', 'drums', 'rest', 'vocals']  # to check that has the same order than the training
    OVERLAP = 0
    MODE = setting(default='conditioned', standard='standard')
    EMB_TYPE = setting(
        default='dense', simple_dense='dense', complex_dense='dense',
        simple_cnn='cnn', complex_cnn='cnn'
    )
