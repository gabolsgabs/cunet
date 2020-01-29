from cunet.preprocess.spectrogram import spec_complex
from tensorflow.keras.models import load_model
import numpy as np
from cunet.evaluation.config import config


def separate_audio_replicate(model, path_audio):
    config.set_group('standard')
    from cunet.evaluation.results import analize_spec
    y, _ = analize_spec(
        spec_complex(path_audio)['spec'], model, None)
    y = (y - np.min(y))/(np.max(y) - np.min(y))
    # name = path_audio.split('/')[-1].replace('.mp3', '.wav')
    # name = os.path.join(path_output, name)
    # librosa.output.write_wav(name, y, sr=config_prepro.FR)
    return


def load_a_cunet():
    model = load_model('cunet/evaluation/with_val_all_files.h5')
    return model
