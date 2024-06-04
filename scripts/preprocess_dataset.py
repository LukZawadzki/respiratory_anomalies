import argparse
import os
import librosa
import numpy as np
import soundfile


def _get_arg_parser():

    parser = argparse.ArgumentParser(description='Preprocess the dataset.')

    parser.add_argument('--dataset_folder', type=str,
                        required=True, help='Path to the dataset folder.')
    parser.add_argument('--shift_max', type=int, default=2000,
                        help='Maximum shift in time domain.')
    parser.add_argument('--min_clipping', type=float, default=0.01,
                        help='Minimum clipping in frequency domain.')
    parser.add_argument('--max_clipping', type=float, default=0.01,
                        help='Maximum clipping in frequency domain.')
    parser.add_argument('--extend_to_three_channels', type=bool, default=False,
                        help='Whether to extend the spectrogram to three channels.')
    parser.add_argument('--output_folder', type=str,
                        required=True, help='Path to the output folder.')

    return parser


def _load_audio(file_path, target_sr=22050):
    """Loads the audio file from the given path."""

    audio, sr = librosa.load(file_path, sr=None)

    if sr != target_sr:
        audio = librosa.resample(audio, target_sr=target_sr, orig_sr=sr)

    return audio


def _get_audio_parts(path, sr):
    """Returns the audio parts and their labels from the given file."""

    audio = _load_audio(path, sr)

    with open(path.replace('.wav', '.txt'), 'r') as meta_file:

        for line in meta_file:

            start, end = line.split()[:2]
            start, end = int(float(start) * sr), int(float(end) * sr)

            has_crackles = int(line.split()[2])
            has_wheezes = int(line.split()[3])

            label = 2 * has_crackles + has_wheezes

            yield audio[start:end], label


def _random_time_shift(audio, shift_max):
    """Randomly shifts the audio in time domain."""

    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(audio, shift)


def _preprocess_audio(audio, shift_max, min_clipping, max_clipping, extend_to_three_channels, target_size, sr):

    # pad the audio to the maximum length
    audio = audio[:target_size]
    audio = np.pad(audio, (0, target_size - len(audio)))

    # random time shift
    audio = _random_time_shift(audio, shift_max)

    # create the spectrogram
    audio = np.abs(librosa.stft(audio, n_fft=1024, hop_length=512))
    audio = librosa.amplitude_to_db(audio, ref=np.max)
    audio += abs(np.min(audio))
    audio = audio[:64, :128]

    # random spectral clipping
    clipped_part = np.random.uniform(min_clipping, max_clipping)
    clipped_mels = int(clipped_part * len(audio))
    clipped_start = np.random.randint(0, len(audio) - clipped_mels)
    audio[clipped_start:clipped_start + clipped_mels, :] = 0.0

    audio /= np.max(audio)

    audio = np.expand_dims(audio, 2)

    # extend to three channels
    if extend_to_three_channels:
        audio = np.concatenate([audio, audio, audio], 2)

    return audio


def main(args: argparse.Namespace):
    """Loads the dataset, preprocess the files and writes them to a new folder."""

    file_paths = [os.path.join(args.dataset_folder, path) for path in os.listdir(
        args.dataset_folder) if path.endswith('.wav')]

    sampling_rate = 22050
    target_size = sampling_rate * 3

    out_file_index = 0

    for _ in range(5):

        for file_path in file_paths:

            for (audio, label) in _get_audio_parts(file_path, sampling_rate):

                output_path = os.path.join(
                    args.output_folder, f'file_{out_file_index}_{label}.npy')
                
                # wav_path = output_path.replace('.npy', '.wav')

                # soundfile.write(wav_path, audio, sampling_rate)
                
                audio = _preprocess_audio(audio, args.shift_max, args.min_clipping,
                                          args.max_clipping, args.extend_to_three_channels, target_size, sampling_rate)

                np.save(output_path, audio)

                out_file_index += 1


if __name__ == '__main__':

    parser = _get_arg_parser()
    main(parser.parse_args())
