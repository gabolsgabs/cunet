# -*- coding: utf-8 -*-
from effortless_config import Config


class config(Config):
    # PATH_MODEL = '/net/guzheng/data2/anasynth_nonbp/meseguerbrocal/source_separation/musdb18/models/standard/vocals/model.h5'
    PATH_MODEL = '/net/guzheng/data2/anasynth_nonbp/meseguerbrocal/source_separation/musdb18/models/conditioned/simple_dense/test/test_final.h5'
    PATH_AUDIO = '/net/guzheng/data2/anasynth_nonbp/meseguerbrocal/source_separation/musdb18/test/complex'
    TARGET = ['vocals']  # ['vocals', 'bass', 'bass_vocals'] -> not ready yet for complex conditions
    INSTRUMENTS = ['bass', 'drums', 'rest', 'vocals']  # to check that has the same order than the training
    OVERLAP = 0
    PATH_RESULTS = "/".join(PATH_MODEL.split('/')[:-1])
    if 'standard' in PATH_MODEL:
        MODE = 'standard'
    if 'conditioned'in PATH_MODEL:
        MODE = 'conditioned'
        EMB_TYPE = PATH_MODEL.split('/')[-3].split('_')[-1]
