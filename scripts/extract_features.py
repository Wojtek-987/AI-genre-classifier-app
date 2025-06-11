import os
import argparse
import random

import numpy as np
import librosa


def maybe_augment(y, sr):
    # Randomly apply either pitch-shift or time-stretch, or return y unchanged.
    r = random.random()
    if r < 0.5:
        # pitch-shift ±1 semitone
        n_steps = random.choice([-1, 1])
        return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    elif r < 0.75:
        # time-stretch ±10%
        rate = random.uniform(0.9, 1.1)
        return librosa.effects.time_stretch(y=y, rate=rate)
    else:
        # no augmentation
        return y


def extract_and_save(src_dir, dst_dir, n_mels, aug_per_file):
    os.makedirs(dst_dir, exist_ok=True)
    for genre in sorted(os.listdir(src_dir)):
        src_g = os.path.join(src_dir, genre)
        dst_g = os.path.join(dst_dir, genre)
        if not os.path.isdir(src_g):
            continue
        os.makedirs(dst_g, exist_ok=True)

        for fname in sorted(os.listdir(src_g)):
            if not fname.lower().endswith('.wav'):
                continue
            base, _ = os.path.splitext(fname)
            in_path = os.path.join(src_g, fname)

            # Load original audio once
            y_orig, sr = librosa.load(in_path, sr=None)

            # Always save the original Mel
            S = librosa.feature.melspectrogram(y=y_orig, sr=sr, n_mels=n_mels)
            S_dB = librosa.power_to_db(S, ref=np.max)
            out_name = f"{base}_orig.npy"
            out_path = os.path.join(dst_g, out_name)
            np.save(out_path, S_dB)
            print(f"✔ {genre}/{fname} → {out_name}")

            # Generate N augmentations
            for i in range(1, aug_per_file + 1):
                y_aug = maybe_augment(y_orig, sr)
                S_aug = librosa.feature.melspectrogram(y=y_aug, sr=sr, n_mels=n_mels)
                S_aug_dB = librosa.power_to_db(S_aug, ref=np.max)
                aug_name = f"{base}_aug{i}.npy"
                aug_path = os.path.join(dst_g, aug_name)
                np.save(aug_path, S_aug_dB)
                print(f"  ↳ aug #{i} saved as {aug_name}")


def main():
    p = argparse.ArgumentParser(
        description="Extract Mel-spectrograms and augment them"
    )
    p.add_argument(
        '--src', default='data/genres',
        help="Source folder of genre subdirs"
    )
    p.add_argument(
        '--dst', default='data/features',
        help="Destination for .npy feature files"
    )
    p.add_argument(
        '--n_mels', type=int, default=128,
        help="Number of Mel bands"
    )
    p.add_argument(
        '--aug_per_file', type=int, default=1,
        help="How many augmented versions to generate per WAV"
    )
    args = p.parse_args()
    random.seed(42)
    extract_and_save(args.src, args.dst, args.n_mels, args.aug_per_file)


if __name__ == "__main__":
    main()
