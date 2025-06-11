"""
Command-line tool to evaluate a saved Keras genre classifier.
Loads mel-spectrogram features, runs inference, computes metrics,
and writes results to CSV and a confusion matrix image.

Usage:

  # evaluate using the default checkpoint (best_model.keras in model_dir)
  python scripts/evaluate.py `
    --features data/features `
    --model_dir models/run_best_test `
    --output  models/run_best_test/eval_best_test.csv

  # evaluate using the final checkpoint explicitly
  python scripts/evaluate.py `
    --features   data/features `
    --model_dir  models/run_best_test `
    --checkpoint models/run_best_test/model_final.keras `
    --output     models/run_best_test/eval_final.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def discover_genres(feature_dir):
    # subfolders of feature_dir define genre labels
    return sorted([
        d for d in os.listdir(feature_dir)
        if os.path.isdir(os.path.join(feature_dir, d))
    ])


def load_all_features(feature_dir, genres):
    # gather all (genre, filename) pairs
    entries = []
    for idx, genre in enumerate(genres):
        gd = os.path.join(feature_dir, genre)
        for fname in sorted(os.listdir(gd)):
            if fname.endswith('.npy'):
                entries.append((idx, genre, fname))

    if not entries:
        raise RuntimeError(f"No .npy files found under {feature_dir}")

    # infer target frame length from first sample
    _, first_genre, first_file = entries[0]
    mel0 = np.load(os.path.join(feature_dir, first_genre, first_file))
    if mel0.ndim == 2:
        mel0 = mel0[..., np.newaxis]
    target_frames = mel0.shape[1]

    X_list, y_list, paths = [], [], []
    for label, genre, fname in entries:
        arr = np.load(os.path.join(feature_dir, genre, fname))
        # ensure 3D (H, W, 1)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]

        # trim or pad to target_frames
        frames = arr.shape[1]
        if frames > target_frames:
            arr = arr[:, :target_frames, :]
        elif frames < target_frames:
            pad_amt = target_frames - frames
            # pad only in time dimension
            arr = np.pad(arr, ((0, 0), (0, pad_amt), (0, 0)), mode='constant')

        X_list.append(arr)
        y_list.append(label)
        paths.append(os.path.join(genre, fname))

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    return X, y, paths


def batch_predict(model, X, batch_size=32):
    # run predictions in batches
    preds = []
    n = X.shape[0]
    for start in range(0, n, batch_size):
        batch = X[start:start + batch_size]
        p = model.predict(batch, verbose=0)
        preds.append(p)
    return np.vstack(preds)


def plot_and_save_confusion(cm, genres, out_png):
    # render and save confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(np.arange(len(genres)), genres, rotation=45, ha='right')
    plt.yticks(np.arange(len(genres)), genres)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved confusion matrix: {out_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Keras genre‐classifier"
    )
    parser.add_argument(
        '--features', required=True,
        help="Path to data/features/"
    )
    parser.add_argument(
        '--model_dir', required=True,
        help="Directory with a Keras checkpoint"
    )
    parser.add_argument(
        '--checkpoint', default=None,
        help="Explicit checkpoint path (e.g. model_final.keras)"
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        '--output', default=None,
        help="CSV path for saving metrics"
    )
    args = parser.parse_args()

    # choose checkpoint
    ckpt = args.checkpoint or os.path.join(args.model_dir, 'best_model.keras')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model = load_model(ckpt)
    print(f"Loaded model from {ckpt}")

    # get genre list
    genres = discover_genres(args.features)
    print(f"Found genres: {genres}")

    # load and standardise features
    X, y_true, paths = load_all_features(args.features, genres)
    print(f"Loaded {len(paths)} samples")

    # predict and evaluate
    pred_proba = batch_predict(model, X, batch_size=args.batch_size)
    y_pred = np.argmax(pred_proba, axis=1)

    # compile metrics
    metrics = {
        'set': ['all_data'],
        'accuracy': [accuracy_score(y_true, y_pred)],
        'precision': [precision_score(y_true, y_pred, average='macro', zero_division=0)],
        'recall': [recall_score(y_true, y_pred, average='macro', zero_division=0)],
        'f1': [f1_score(y_true, y_pred, average='macro', zero_division=0)]
    }
    df = pd.DataFrame(metrics)
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Metrics saved to {args.output}")
    else:
        print(df)

    # confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    out_png = os.path.join(args.model_dir, 'confusion_matrix.png')
    plot_and_save_confusion(cm, genres, out_png)


if __name__ == "__main__":
    main()
