"""
Custom Keras Sequence that loads mel-spectrogram features by genre.
Searches a feature directory for .npy files, assigns labels,
and pads or trims each sample to a fixed frame length.
"""

import os
import glob
import random

import numpy as np
from tensorflow.keras.utils import Sequence


class MelDataset(Sequence):
    def __init__(self, feature_dir, genres, batch_size=32, shuffle=True, max_frames=None):
        """
        feature_dir : root folder containing genre subfolders
        genres      : list of genre names, e.g. ['blues','classical',…]
        batch_size  : number of samples per batch
        shuffle     : whether to shuffle samples each epoch
        max_frames  : fixed number of time frames (inferred if None)
        """
        self.feature_dir = feature_dir
        self.genres = genres
        self.batch_size = batch_size
        self.shuffle = shuffle

        # collect all (file_path, label) pairs
        self.samples = []
        for label_idx, genre in enumerate(self.genres):
            pattern = os.path.join(self.feature_dir, genre, '**', '*.npy')
            for filepath in glob.glob(pattern, recursive=True):
                self.samples.append((filepath, label_idx))

        if not self.samples:
            print(f"Warning: no .npy files found in {self.feature_dir} for genres {self.genres}")

        # determine max_frames if not provided
        if max_frames is None and self.samples:
            first_mel = np.load(self.samples[0][0])
            self.max_frames = first_mel.shape[1]
        else:
            self.max_frames = max_frames or 0

        # prepare for the first epoch
        self.on_epoch_end()

    def __len__(self):
        # number of batches per epoch
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, idx):
        # fetch batch of samples
        batch_samples = self.samples[
                        idx * self.batch_size: (idx + 1) * self.batch_size
                        ]
        X, y = [], []

        for filepath, label in batch_samples:
            mel = np.load(filepath)
            n_mels, frames = mel.shape

            # trim longer clips or pad shorter ones
            if frames > self.max_frames:
                mel = mel[:, :self.max_frames]
            elif frames < self.max_frames:
                pad_width = self.max_frames - frames
                mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')

            # add channel dimension for CNN input
            X.append(mel[..., np.newaxis])
            y.append(label)

        return np.stack(X, axis=0), np.array(y)

    def on_epoch_end(self):
        # shuffle samples at epoch end if required
        if self.shuffle:
            random.shuffle(self.samples)
