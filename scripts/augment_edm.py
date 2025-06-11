"""
Go through every WAV in ../data/genres/edm and create three tweaks of each:
  1) time-stretch around ±15%
  2) pitch-shift by up to ±2 semitones
  3) add Gaussian noise (0.3% max amp)

Files already tagged with "_aug" are skipped to avoid looping over them again.
"""

import os
import librosa
import numpy as np
import soundfile as sf

EDM_DIR = "../data/genres/edm"
VARIANTS_PER_TRACK = 3


def make_variants(y: np.ndarray, sr: int):
    variants = []
    # 1) Stretch the timing by a random factor between 0.85 and 1.15
    stretch_rate = np.random.uniform(0.85, 1.15)
    variants.append(librosa.effects.time_stretch(y, rate=stretch_rate))

    # 2) Shift pitch by a random integer in [-2, 2]
    semitone_shift = np.random.randint(-2, 3)
    variants.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=semitone_shift))

    # 3) Add Gaussian noise (0.3% of peak amplitude)
    noise = np.random.randn(len(y)) * 0.003 * np.max(np.abs(y))
    variants.append(y + noise)

    return variants


def main():
    for fname in sorted(os.listdir(EDM_DIR)):
        if not fname.lower().endswith(".wav"):
            continue
        if "_aug" in fname:
            # already processed
            continue

        infile = os.path.join(EDM_DIR, fname)
        y, sr = librosa.load(infile, sr=None, mono=True)
        base_name, _ = os.path.splitext(fname)

        # craft the variants
        tracks = make_variants(y, sr)[:VARIANTS_PER_TRACK]
        for idx, y_var in enumerate(tracks):
            out_fname = f"{base_name}_aug{idx}.wav"
            outfile = os.path.join(EDM_DIR, out_fname)
            if os.path.exists(outfile):
                # don't overwrite existing file
                continue
            sf.write(outfile, y_var, sr)
            print(f"✓ Saved variant: {out_fname}")


if __name__ == "__main__":
    main()
