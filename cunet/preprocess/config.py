# -*- coding: utf-8 -*-
from effortless_config import Config, setting


class config(Config):
    PATH_RAW = '/data2/anasynth_nonbp/meseguerbrocal/source_separation/musdb18/test/raw_audio/'
    PATH_SPEC = '/data2/anasynth_nonbp/meseguerbrocal/source_separation/musdb18/test/complex/'
    FR = 8192
    FFT_SIZE = 1024
    HOP = 256
    N_FFT = None
    INTRUMENTS = ['bass', 'drums', 'rest', 'vocals', 'mix']
    CONDITIONS = ['bass', 'drums', 'rest', 'vocals']
    COMPLEX = 1     # complex conditions -> 1 only original instrumets, 2 combination of 2 instruments, etc
    ADD_ZERO = True    # add the zero condition
    ADD_ALL = True     # add the all mix condition
    ADD_IN_BETWEEN = [1.]   # in between values for the combination of several instruments
    CHUNK_SIZE = 1
