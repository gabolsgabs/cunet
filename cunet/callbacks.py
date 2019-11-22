from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, TensorBoard
)
from cunet.config import config


def make_earlystopping(patience):
    return EarlyStopping(
        monitor='val_loss', min_delta=0, mode='min', patience=patience,
        verbose=1, restore_best_weights=True
    )


def make_checkpoit(name_ckpt):
    return ModelCheckpoint(
        name_ckpt, verbose=1, mode='min', save_best_only=True,
        monitor='val_loss'
    )


def make_tensorboard():
    return TensorBoard(log_dir=config.PATH_LOG, write_graph=True)


# saved_model_callback = SavedModelCallback(
#     save_path=os.path.join(saved_model_dir, 'model'),
#     serving_only=True,
#     verbose=1,
# )
# checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=os.path.join(checkpoint_dir, 'model'),
#     save_best_only=True,
#     monitor='val_loss',
#     verbose=1,
#     save_weights_only=True
# )
#
# visualize_callback = VisualizeCallback(
#     model, train_ds, test_ds, tensorboard_dir)
# early_stopping_callback = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     min_delta=config.EARLY_STOPPING_MIN_DELTA,
#     patience=config.EARLY_STOPPING_PATIENCE,
#     verbose=1,
# )
# reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1
# )
