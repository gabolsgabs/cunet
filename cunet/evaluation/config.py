# -*- coding: utf-8 -*-
from effortless_config import Config, setting


class config(Config):
    MODE = 'standard'
    PATH_MODEL = ''
    PATH_AUDIO = '/data2/anasynth_nonbp/meseguerbrocal/source_separation/musdb18/test/complex'
    PATH_RESULTS = ''
    SOURCE = ['vocals']
    INSTRUMENTS = ['bass', 'drums', 'rest', 'vocals']  # to check that has the same order than the training
    EMB_TYPE = 'dense'
    OVERLAP = 0
