import scipy
import numpy as np
from glob import glob
from cunet.preprocess.config import config
import logging
import os
import subprocess as sp


def read_MP3(audio, sr_hz=16000., stereo2mono=False):
    """Transform any audio format into mp3 - ALICE VERSION"""
    if os.path.isfile(audio):
        if (
            audio.endswith('mp3') | audio.endswith('aif')
            | audio.endswith('aiff') | audio.endswith('wav')
            | audio.endswith('ogg') | audio.endswith('flac')
        ):
            # --- resample and reduce to mono
            if stereo2mono:
                ffmpeg = sp.Popen(
                    ["ffmpeg", "-i", audio, "-vn", "-acodec",
                     "pcm_s16le", "-ac", "1", "-ar", str(sr_hz), "-f", "s16le",
                     "-"], stdin=sp.PIPE, stdout=sp.PIPE,
                    stderr=open(os.devnull, "w"))
            else:
                # --- resample and keep stereo
                ffmpeg = sp.Popen(
                    ["ffmpeg", "-i", audio, "-vn", "-acodec",
                     "pcm_s16le", "-ac", "2", "-ar", str(sr_hz), "-f",
                     "s16le", "-"], stdin=sp.PIPE, stdout=sp.PIPE,
                    stderr=open(os.devnull, "w"))
            raw_data = ffmpeg.stdout
            mp3_array = np.frombuffer(raw_data.read(), np.int16)
            mp3_array = mp3_array.astype('float32') / 32767.0
            data_v = mp3_array.view()
            if not stereo2mono:
                data_v = np.reshape(data_v, (int(data_v.shape[0]/2), 2))
            return data_v, sr_hz


def stft(x, fft_size=1024, hop=768, n_fft=None):
    """time_r = hop/sr_hz"""
    w = scipy.hanning(fft_size+1)[:-1]
    size_spec = int((x.shape[0]-fft_size)/hop + 1)
    return np.array([np.fft.rfft(w*x[i*hop:i*hop+fft_size], n_fft)
                     for i in range(0, size_spec)]).astype(np.complex64)


def spec_complex(audio_file):
    """Compute the complex spectrum"""
    output = {'type': 'complex'}
    logger = logging.getLogger('computing_spec')
    try:
        logger.info('Computing complex spec for %s' % audio_file)
        audio, fe = read_MP3(audio_file, stereo2mono=True, sr_hz=config.FR)
        output['spec'] = stft(
            audio, fft_size=config.FFT_SIZE, hop=config.HOP,
            n_fft=config.N_FFT
        ).T
    except Exception as my_error:
        logger.error(my_error)
    return output


def spec_mag(audio_file, norm=True):
    """Compute the normalized mag spec and the phase of an audio file"""
    output = {}
    logger = logging.getLogger('computing_spec')
    try:
        spec, error = spec_complex(audio_file)
        spec = spec['spec']
        logger.info('Computing mag and phase for %s' % audio_file)
        # n_freq_bins -> connected with fft_size with 1024 -> 513 bins
        # the number of band is odd -> removing the last band
        n = spec.shape[1] - 1
        mag = np.abs(spec[:, :n])
        #  mag = mag / np.max(mag)
        if norm:
            mx = np.max(mag)
            mn = np.min(mag)
            #  betweens 0 and 1 (x - min(x)) / (max(x) - min(x))
            mag = ((mag - mn) / (mx-mn))
            output['norm_param'] = np.array([mx, mn])
        output['phase'] = np.angle(spec)
        output['magnitude'] = mag
    except Exception as my_error:
        logger.error(my_error)
    return output


def spec_mag_log(audio_file):
    """Compute the normalized log mag spec and the phase of an audio file"""
    output = {}
    logger = logging.getLogger('computing_spec')
    try:
        tmp, error = spec_mag(audio_file, False)    # mag without norm
        mag = tmp['magnitude']
        output['phase'] = tmp['phase']
        spec_log = np.log1p(mag)
        mx = np.max(spec_log)
        mn = np.min(spec_log)
        output['norm_param'] = np.array([mx, mn])
        output['log_magnitude'] = (spec_log - mn) / (mx - mn)
    except Exception as my_error:
        logger.error(my_error)
    return output


def compute_one_song(folder):
    logger = logging.getLogger('computing_spec')
    name = os.path.basename(os.path.normpath(folder)).replace(' ', '_')
    logger.info('Computing spec for %s' % name)
    data = {i: spec_complex(folder+i+'.wav')['spec']
            for i in config.INTRUMENTS}
    np.savez(os.path.join(config.PATH_SPEC, name+'.npz'), **data)
    return


def main():
    logging.basicConfig(
        filename=os.path.join(config.PATH_SPEC, 'computing_spec.log'),
        level=logging.INFO
    )
    logger = logging.getLogger('computing_spec')
    logger.info('Starting the computation')
    for i in glob(os.path.join(config.PATH_RAW, '*/')):
        if os.path.isdir(i):
            compute_one_song(i)
    return


if __name__ == '__main__':
    main()
