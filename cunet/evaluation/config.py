# -*- coding: utf-8 -*-
from effortless_config import Config, setting


class config(Config):
    MODE = 'standard'
    PATH_MODEL = '/net/guzheng/data2/anasynth_nonbp/meseguerbrocal/source_separation/musdb18/models/standard/vocals/model.h5'
    PATH_AUDIO = '/net/guzheng/data2/anasynth_nonbp/meseguerbrocal/source_separation/musdb18/test/complex'
    PATH_RESULTS = ''
    TARGET = ['vocals']
    INSTRUMENTS = ['bass', 'drums', 'rest', 'vocals']  # to check that has the same order than the training
    EMB_TYPE = 'dense'
    OVERLAP = 0
