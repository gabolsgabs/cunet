# -*- coding: utf-8 -*-
from effortless_config import Config, setting


class config(Config):
    groups = ['train', 'test']
    PATH_BASE = '/net/guzheng/data2/anasynth_nonbp/meseguerbrocal/source_separation/musdb18/'
    PATH_RAW = setting(
        default=PATH_BASE+'train/raw_audio/',
        train=PATH_BASE+'train/raw_audio/', test=PATH_BASE+'test/raw_audio/'
    )
    PATH_SPEC = setting(
        default=PATH_BASE+'train/complex/',
        train=PATH_BASE+'train/complex/', test=PATH_BASE+'test/complex/'
    )
    PATH_INDEXES = setting(
        default=PATH_BASE+'train/indexes/',
        train=PATH_BASE+'train/indexes/', test=PATH_BASE+'test/indexes/'
    )
    FR = 8192
    FFT_SIZE = 1024
    HOP = 256
    MODE = 'conditioned'
    INTRUMENTS = ['bass', 'drums', 'rest', 'vocals', 'mix']
    CONDITIONS = ['bass', 'drums', 'rest', 'vocals']
    CONDITION_MIX = 1       # complex conditions -> 1 only original instrumets, 2 combination of 2 instruments, etc
    ADD_ZERO = True         # add the zero condition
    ADD_ALL = True          # add the all mix condition
    ADD_IN_BETWEEN = 1.     # in between interval for the combination of several instruments
    STEP = 1                # step in frames for creating the input data
    CHUNK_SIZE = 4          # chunking the indexes before mixing -> define the number of data points of the same track
