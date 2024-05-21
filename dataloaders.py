import os

import tensorflow as tf
import librosa
import numpy as np


def _load_audio(file_path):
    # Load the audio file using librosa
    audio, sr = librosa.load(file_path.numpy(), sr=None, mono=True)

    if sr != 44100:
        # print("Resampling from ", sr, " to 44100")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
        sr = 44100

    audio = audio[:44100 * 20]

    if len(audio) < 44100 * 20:
        audio = np.pad(audio, (0, 44100 * 20 - len(audio)))

    return audio, sr

def _random_time_shift(audio, shift_max):
    # Randomly shift the audio by up to shift_max samples
    # shift = np.random.randint(-shift_max, shift_max)
    # return np.roll(audio, shift)
    return audio

def _random_spectral_clipping(audio, min_clipping, max_clipping):
    # Randomly apply spectral clipping
    # threshold = np.random.uniform(min_clipping, max_clipping)
    # clipped_audio = np.clip(audio, -threshold, threshold)
    # return clipped_audio
    return audio


def _compute_specgram(audio, sr):
    # Compute the spectrogram of the audio
    specgram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, hop_length=1024, n_fft=2048)
    specgram = librosa.power_to_db(specgram, ref=np.max)
    return specgram

def _preprocess_audio(file_path, shift_max, min_clipping, max_clipping, extend_to_three_channels):
    
    def func(fp, smax, minc, maxc, extend):
        audio, sr = _load_audio(fp)
        audio = _random_time_shift(audio, smax)
        audio = _random_spectral_clipping(audio, minc, maxc)
        audio = _compute_specgram(audio, sr)[:, :512]
        audio = np.expand_dims(audio, 2)

        if extend:
            audio = np.concatenate([audio, audio, audio], 2)

        return audio
    
    return tf.py_function(func, [file_path, shift_max, min_clipping, max_clipping, extend_to_three_channels], Tout=tf.float32)


def _get_label(file_path):
    file_descriptor = file_path.replace('.wav', '.txt')

    count_crackles = 0
    count_wheezes = 0

    with open(file_descriptor, 'r') as f:
        for line in f:
            split_line = line.split()

            if split_line[2] == '1':
                count_crackles += 1

            if split_line[3] == '1':
                count_wheezes += 1

    if count_crackles > 0 and count_wheezes > 0:
        return 3

    if count_crackles > 0:
        return 2

    if count_wheezes > 0:
        return 1

    return 0


class DataSetLoader:
    """Loads the dataset and preprocesses the audio files."""

    def __init__(self, dataset_folder, shift_max, min_clipping, max_clipping, extend_to_three_channels=False):

        self._dataset_folder = dataset_folder
        self._shift_max = shift_max
        self._min_clipping = min_clipping
        self._max_clipping = max_clipping
        self._extend_to_three_channels = extend_to_three_channels

    def create_dataset(self, batch_size, train_split=0.8):

        file_paths = [os.path.join(self._dataset_folder, path) for path in os.listdir(
            self._dataset_folder) if path.endswith('.wav')]
        labels = [_get_label(path) for path in file_paths]

        file_paths = tf.data.Dataset.from_tensor_slices(file_paths)
        dataset = file_paths.map(lambda file_path: _preprocess_audio(file_path, self._shift_max, self._min_clipping, self._max_clipping, self._extend_to_three_channels),
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = tf.data.Dataset.zip((dataset, tf.data.Dataset.from_tensor_slices(labels)))

        dataset = dataset.batch(batch_size)
        # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset.take(int(len(file_paths) * train_split)), dataset.skip(int(len(file_paths) * train_split))
