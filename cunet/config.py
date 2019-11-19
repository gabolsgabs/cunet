# -*- coding: utf-8 -*-
from effortless_config import Config, setting


class config(Config):
    PATH_INPUT = ''
    PATH_OUTPUT = ''
    PATH_LOG = ''
    TARGET = 'vocals'   # only for standard version
    BATCH_SIZE = 32
    N_BATCH = 256
