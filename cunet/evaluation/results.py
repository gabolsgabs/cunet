from glob import glob
import librosa
from tensorflow.keras.models import load_model
import mir_eval
import numpy as np
import os
import logging
from cunet.evaluation import config


def istft(data, data_config):
    return librosa.istft(
        data, hop_length=data_config.item()['HOP'],
        win_length=data_config.item()['FFT_SIZE'])


def prepare_a_song(spec, N, num_bands):
    size = spec.shape[0]
    segments = np.zeros((size//(N-config.OVERLAP)+1, num_bands, N, 1),
                        dtype=np.float32)
    for index, i in enumerate(np.arange(0, size, N-config.OVERLAP)):
        segment = spec[:num_bands, i:i+N]
        tmp = segment.shape[1]
        if tmp != N:
            segment = np.zeros((num_bands, N), dtype=np.float32)
            segment[:, :tmp] = spec[i:i+N]
        segments[index] = np.expand_dims(np.abs(segment), axis=2)
    return segments


def separate_audio(path_audio, path_output, model, params, name=None):
    pred_audio, pred_spec = analize_spec(path_audio, model, **params)
    if not name:
        name = path_audio.split('/')[-1].replace('.mp3', '.wav')
    name = os.path.join(path_output, name)
    librosa.output.write_wav(pred_audio, name)
    return


def concatenate(data):
    output = np.array([])
    if config.OVERLAP == 0:
        output = np.concatenate(data, axis=1)
    else:
        output = data[0]
        o = int(config.OVERLAP/2)
        f = 0
        if config.OVERLAP % 2 != 0:
            f = 1
        for i in range(1, data.shape[0]):
            output = np.concatenate(
                (output[:, :-(o+f), :], data[i][:, o:, :]), axis=1)
    return output


def analize_spec(spec, model, data_config):
    logger = logging.getLogger('computing_spec')
    pred_audio = np.array([])
    pred_spec = np.array([])
    try:
        num_bands, N = model.input_shape[1:3]
        x = prepare_a_song(spec, N, num_bands)
        if config.MODE == 'standard':
            pred_spec = model.predict(x)
        if config.MODE == 'conditioned':
            cond = np.zeros(len(config.INSTRUMENTS))
            for i in config.SOURCE:
                cond[config.INSTRUMENTS.index(i)] = 1.
            if config.EMB_TYPE == 'dense':
                cond = cond.reshape(1, -1)
            if config.EMB_TYPE == 'cnn':
                cond = cond.reshape(-1, 1)
            tmp = np.zeros((x.shape[0], *cond.shape))
            tmp[:] = cond
            pred_spec = model.predict([x, tmp])
        pred_spec = np.transpose(
            np.squeeze(concatenate(pred_spec), axis=-1))[:spec.shape[1], :]
        pred_audio = istft(pred_spec, data_config)
    except Exception as my_error:
        logger.error(my_error)
    return pred_audio, pred_spec


def do_an_exp(fl, model):
    audio = np.load(fl, allow_pickle=True)
    accompaniment = np.zeros([1])
    for i in config.INSTRUMENTS:
        if i not in config.SOURCE:
            if len(accompaniment) == 1:
                accompaniment = audio[i]
            else:
                accompaniment = np.sum([accompaniment, audio[i]], axis=0)
    # original isolate source
    source = istft(audio[config.SOURCE], audio['config'])
    # original mix
    mix = istft(audio['mix'], audio['config'])
    # accompaniment (sum of all apart from the original)
    acc = istft(accompaniment, audio['config'])
    # predicted separation
    pred_audio, pred_spec = analize_spec(audio['mix'], model, audio['config'])
    # size
    sz = min(pred_audio.shape[0], source.shape[0], mix.shape[0], acc.shape[0])
    pred_acc = mix[:sz] - pred_audio[:sz]
    pred = np.array([pred_audio[:sz], pred_acc])
    orig = np.array([source[:sz], acc[:sz]])
    """
    librosa.output.write_wav(pred_audio, config.PATH_RESULTS + 'audio/pred_audio.wav')
    librosa.output.write_wav(pred_acc, config.PATH_RESULTS + 'audio/pred_acc.wav')
    """
    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(
        reference_sources=orig, estimated_sources=pred,
        compute_permutation=False
    )
    return sdr[perm[0]], sir[perm[0]], sar[perm[0]]


def get_stats(dict, stat):
    logger = logging.getLogger('computing_spec')
    values = np.fromiter(dict.values(), dtype=float)
    r = {'mean': np.mean(values), 'std': np.std(values)}
    logger.info(stat + " : mean {}, std {}".format(r['mean'], r['std']))
    return r


def main(orig_label, params):
    logging.basicConfig(
        filename=os.path.join(config.PATH_RESULTS, 'computing_spec.log'),
        level=logging.INFO
    )
    logger = logging.getLogger('computing_spec')
    logger.info('Starting the computation')
    files = glob(os.path.join(config.PATH_AUDIO, '*.npz'))
    sdr, sir, sar = {}, {}, {}
    model = load_model(config.PATH_MODEL)
    # model = load_model(path_model, custom_objects={"tf": tf})
    for i, f in enumerate(files):
        name = os.path.basename(os.path.normpath(f)).replace('.npz', '')
        logger.info('Analyzing ' + name)
        logger.info('Song num: ' + str(i+1) + ' out of ' + str(len(files)))
        sdr[name], sir[name], sar[name] = do_an_exp(f, model)
        logger.info(sdr[name], sir[name], sar[name])
    sdr.update(get_stats(sdr, 'SDR'))
    sir.update(get_stats(sir, 'SIR'))
    sar.update(get_stats(sar, 'SAR'))
    np.savez(
        os.path.join(config.PATH_RESULTS, name+'.npz'),
        config=str(config), sdr=sdr, sir=sir, sar=sar
    )
    return


if __name__ == '__main__':
    main()
