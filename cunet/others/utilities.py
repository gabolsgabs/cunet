from datetime import datetime
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
)
from cunet.config import config
import os


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return


def write_config(name):
    folder = os.path.join(save_dir('logs'), name)
    create_folder(folder)
    fl = open(os.path.join(folder, 'config.txt'), 'w')
    fl.write(str(config))
    fl.close()
    return


def save_dir(t):
    folder = os.path.join(config.PATH_BASE, t, config.MODE)
    if config.MODE == 'standard':
        folder = os.path.join(folder, config.TARGET)
    if config.MODE == 'conditioned':
        folder = os.path.join(
            folder, "_".join((config.CONTROL_TYPE, config.FILM_TYPE)))
    create_folder(folder)
    return folder


def make_name():
    now = datetime.now()
    return "_".join((config.NAME, now.strftime("%d_%m_%Y_%H:%M")))


def make_earlystopping():
    return EarlyStopping(
        monitor='val_loss', min_delta=config.EARLY_STOPPING_MIN_DELTA,
        mode='min', patience=config.EARLY_STOPPING_PATIENCE,
        verbose=1, restore_best_weights=True
    )


def make_checkpoint(name):
    folder = os.path.join(save_dir('checkpoint'), name)
    create_folder(folder)
    return ModelCheckpoint(
        filepath=os.path.join(folder, 'model_{epoch:02d}-{val_loss:.5f}'),
        verbose=1, mode='min', save_best_only=True, save_weights_only=True,
        monitor='val_loss'
    )


def make_reduce_lr():
    return ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, min_lr=0.0001, 
        patience=config.REDUCE_PLATEAU_PATIENCE, verbose=1
    )



def make_tensorboard(name):
    folder = os.path.join(save_dir('tensorboard'), name)
    create_folder(folder)
    return TensorBoard(log_dir=folder, write_graph=True)


# visualize_callback = VisualizeCallback(
#     model, train_ds, test_ds, tensorboard_dir)
