from cunet.evaluation.results import analize_spec
from cunet.preprocess.spectrogram import spec_complex
from tensorflow.keras.models import load_model


def separate_audio_replicate(model, path_audio):
    pred_audio, _ = analize_spec(spec_complex(path_audio), model)
    # name = path_audio.split('/')[-1].replace('.mp3', '.wav')
    # name = os.path.join(path_output, name)
    return pred_audio


def load_a_cunet():
    model = load_model('with_val_all_files.h5')
    return model
