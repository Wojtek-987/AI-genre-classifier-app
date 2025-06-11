"""
Predict genre using multiple 30 s snippets around the chorus.
Audio is resampled to 22050 Hz, converted to mono, and we extract three
30 s slices—the middle section and the ones immediately before and after.
Each snippet is analysed separately, and results are displayed in descending
confidence order.
"""

import argparse
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Parameters
TARGET_SR = 22050  # sampling rate expected by the CNN
DURATION = 30.0  # length of each snippet in seconds
MODEL_PATH = "models/run_best_test/best_model.keras"
GENRES = [
    "blues", "classical", "country", "disco", "edm",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock"
]


def extract_snippets(y: np.ndarray, sr: int, duration: float):
    """
    Return three 30 s snippets from y:
      1) the segment immediately before the middle
      2) the middle segment
      3) the segment immediately after the middle

    Each snippet is trimmed or loop‐padded to exactly duration.
    """
    target_len = int(sr * duration)
    total_len = len(y)

    if total_len < target_len:
        # audio too short: loop‐pad a single snippet three times
        padded = _pad_loop(y, target_len)
        return [padded, padded, padded]

    # compute start of the central snippet
    mid_start = (total_len - target_len) // 2

    # define the three start positions
    starts = [
        max(0, mid_start - target_len),
        mid_start,
        min(total_len - target_len, mid_start + target_len)
    ]

    snippets = []
    for s in starts:
        segment = y[s: s + target_len]
        if len(segment) < target_len:
            segment = _pad_loop(segment, target_len)
        snippets.append(segment)

    return snippets


def _pad_loop(arr: np.ndarray, length: int):
    """
    Loop‐pad arr until it's exactly length long.
    """
    repeats = length // len(arr)
    rem = length % len(arr)
    padded = np.tile(arr, repeats)
    if rem:
        padded = np.concatenate((padded, arr[:rem]))
    return padded


def extract_mel_spectrogram(y: np.ndarray, sr: int = TARGET_SR, n_mels: int = 128):
    """
    Compute mel-spectrogram dB and reshape for CNN input:
    (1, n_mels, time_steps, 1).
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db[np.newaxis, ..., np.newaxis]


def main():
    parser = argparse.ArgumentParser(
        description="Predict music genre from three chorus-focused snippets"
    )
    parser.add_argument("file", help="Path to an audio file (wav, mp3, etc.)")
    parser.add_argument(
        "--model", default=MODEL_PATH,
        help="Path to the trained Keras model"
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model}")
    model = load_model(args.model)

    print(f"Loading audio: {args.file}")
    y, _ = librosa.load(args.file, sr=TARGET_SR, mono=True)

    print("Extracting snippets around the middle of the track")
    snippets = extract_snippets(y, TARGET_SR, DURATION)

    results = []
    for label, snippet in zip(["before", "middle", "after"], snippets):
        mel_input = extract_mel_spectrogram(snippet)
        probs = model.predict(mel_input, verbose=0)[0]
        idx = np.argmax(probs)
        genre = GENRES[idx]
        confidence = probs[idx] * 100
        results.append((label, genre, confidence))

    # sort by confidence, highest first
    results.sort(key=lambda x: x[2], reverse=True)

    print("\nPredictions (sorted by confidence):")
    for segment, genre, conf in results:
        print(f"{segment.title():>6}: {genre} ({conf:.1f}% confidence)")


if __name__ == "__main__":
    main()
