import os

import tensorflow as tf
import librosa
import numpy as np

def _get_label(path):
    return int(path.split('.')[-2][-1])

def _load_file(path):
    return tf.convert_to_tensor(np.load(path)), _get_label(path)

class DataSetLoader:
    """Loads the dataset and preprocesses the audio files."""

    def __init__(self, dataset_folder):

        self._dataset_folder = dataset_folder

    def create_dataset(self, batch_size, train_split=0.8):

        file_paths = [os.path.join(self._dataset_folder, path) for path in os.listdir(
            self._dataset_folder) if path.endswith('.npy')]

        # labels = [_get_label(path) for path in file_paths]

        dataset_gen = map(_load_file, file_paths)

        dataset = tf.data.Dataset.from_generator(lambda: dataset_gen, output_signature=(tf.TensorSpec(shape=(64, 128, 3), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int32)))

        # dataset = tf.data.Dataset.zip((dataset, tf.data.Dataset.from_tensor_slices(labels)))

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset.take(int(len(file_paths) * train_split)), dataset.skip(int(len(file_paths) * train_split))
