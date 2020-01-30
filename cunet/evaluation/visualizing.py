import tensorflow as tf
import copy
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import plot_model
from cunet.evaluation.config import config
from cunet.evaluation.results import concatenate, reconstruct, spec_complex, prepare_a_song
from cunet.train.load_data_offline import normlize_complex
import numpy as np
import itertools


def run_generic_model(model, model_type, path_audio):
    control_layers = [
        0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 13, 14, 19, 20, 25, 26, 31, 32, 37, 38,
        43, 44
    ]
    shapes = [
        [256, 64, 16], [128, 32, 32], [64, 16, 64],
        [32, 8, 128], [16, 4, 256],     # [8, 2, 512] -> the last one is not concatenated
    ]
    input_spec = spec_complex(path_audio)['spec']
    input_spec = normlize_complex(input_spec)
    input_mag = np.abs(input_spec)
    input_phase = np.angle(input_spec)
    num_bands, num_frames = model.input_shape[0][1:3]
    input_net = prepare_a_song(input_mag, num_frames, num_bands)
    encoder = []
    if model_type == 'complex':
        gammas = np.ones((input_net.shape[0], 1, 512))
        betas = np.zeros((input_net.shape[0], 1, 512))
    if model_type == 'simple':
        gammas = np.ones((input_net.shape[0], 1))
        betas = np.zeros((input_net.shape[0], 1))
    c = 0
    x = input_net
    features = []
    for i, layer in enumerate(model.layers):
        if i not in control_layers:
            if isinstance(layer.input, list):
                # film layer
                if len(layer.input) == 3 and 'lambda' in layer.name:
                    if model_type == 'simple':
                        x = [x, gammas, betas]
                    if model_type == 'complex':
                        s = layer.input[-1].shape[-1]
                        x = [x, gammas[:, :, :s], betas[:, :, :s]]
                # skip connections
                if len(layer.input) == 2:
                    x = [x, encoder[len(encoder) - 1 - c]]
                    c += 1
            x = layer(x)
            features.append(x)
            if (
                layer.output.shape[1:] in shapes
                and 'leaky_re_lu' in layer.name
            ) or i == 7:    # first layer
                encoder.append(x)
    pred_mag = np.squeeze(concatenate(x, input_spec.shape), axis=-1)
    audio = reconstruct(pred_mag, input_phase, input_spec)
    audio = (audio - np.min(audio))/(np.max(audio) - np.min(audio))
    return audio


def get_gammas_betas(model, mix=1):
    conditions = [
        np.array(i).astype(np.float)
        for i in list(itertools.product([0, 1], repeat=4))
        if np.sum(i) <= mix and np.sum(i) > 0
    ]
    # All conditions
    conditions.append(np.ones(4))
    # Zero conditions
    conditions.append(np.zeros(4))
    model_control = Model(
        inputs=model.inputs[1],
        outputs=[model.layers[10].output, model.layers[11].output]
    )

    n_gb = list(model_control.output[0].shape)[-1]
    gammas = np.empty((len(conditions), n_gb))
    betas = np.empty((len(conditions), n_gb))
    for i, c in enumerate(conditions):
        g, b = model_control.predict(c.reshape(1, 1, -1))
        gammas[i, ] = np.squeeze(g)
        betas[i, ] = np.squeeze(b)
    return gammas, betas, conditions


if __name__ == '__main__':
    path_models = '/Users/meseguerbrocal/Desktop/tmp/models/source_separation/'
    version = 'cd_two_cond.h5'
    model = load_model(path_models+version,  custom_objects={"tf": tf})
