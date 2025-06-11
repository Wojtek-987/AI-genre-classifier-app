"""
Train or resume the audio-genre CNN.

- Finds genre labels from subfolders under the features directory.
- Uses MelDataset to stream mel-spectrograms for training and validation.
- Computes balanced class weights to handle any class imbalance.
- Builds a configurable ConvNet or loads an existing checkpoint.
- Compiles with Adam and sparse categorical crossentropy.
- Trains with ModelCheckpoint, EarlyStopping, and CSVLogger.

Command used to train the final model:
    python scripts/train.py `
      --mode train `
      --features data/features `
      --model_dir models/run_best_test `
      --activation selu `
      --batch_size 8 `
      --epochs 8 `
      --filters 16 32 64 `
      --dropouts 0.3 0.3 0.3 0.6
"""

import os
import glob
import argparse
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from data_generator import MelDataset


def build_model(input_shape, num_classes, activation, filters, dropouts):
    model = Sequential()
    # add convolutional blocks
    for i, f in enumerate(filters):
        if i == 0:
            model.add(Conv2D(f, (3, 3), activation=activation,
                             padding='same', input_shape=input_shape))
        else:
            model.add(Conv2D(f, (3, 3), activation=activation, padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2, 2)))
        model.add(Dropout(dropouts[i]))
    # add dense layers
    model.add(Flatten())
    model.add(Dense(256, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropouts[-1]))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train or resume the audio-genre CNN")
    parser.add_argument('--mode', choices=['train', 'resume'], required=True)
    parser.add_argument('--features', required=True,
                        help="Path to data/features/")
    parser.add_argument('--model_dir', required=True,
                        help="Directory in which to save or load model checkpoints")
    parser.add_argument('--checkpoint', default=None,
                        help="When resuming, path to an existing .keras checkpoint")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--filters', type=int, nargs='+', default=[32, 64, 128])
    parser.add_argument('--dropouts', type=float, nargs='+', default=[0.25, 0.25, 0.25, 0.5])
    parser.add_argument('--n_mels', type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    # Find genre labels from folder names
    genres = sorted([
        d for d in os.listdir(args.features)
        if os.path.isdir(os.path.join(args.features, d))
    ])
    num_classes = len(genres)

    # Create data generators
    train_gen = MelDataset(args.features, genres,
                           batch_size=args.batch_size, shuffle=True)
    val_gen = MelDataset(args.features, genres,
                         batch_size=args.batch_size, shuffle=False)

    # Compute class weights
    # try extracting labels from the generator
    try:
        y_train = np.array(train_gen.labels)
    except AttributeError:
        # fallback: list all .npy files on disk
        y_train = []
        for idx, genre in enumerate(genres):
            files = glob.glob(os.path.join(args.features, genre, '*.npy'))
            y_train += [idx] * len(files)
        y_train = np.array(y_train)

    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=y_train
    )
    class_weight_dict = dict(enumerate(weights))
    print("Class weights:", class_weight_dict)

    # Build a new model or load existing checkpoint
    sample_X, _ = train_gen[0]
    input_shape = sample_X.shape[1:]
    if args.mode == 'train':
        model = build_model(input_shape, num_classes,
                            args.activation, args.filters, args.dropouts)
    else:
        if not args.checkpoint:
            raise ValueError("Must pass --checkpoint when resuming")
        model = load_model(args.checkpoint)

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Set up callbacks
    ckpt_best = ModelCheckpoint(
        filepath=os.path.join(args.model_dir, 'best_model.keras'),
        save_best_only=True, monitor='val_accuracy', mode='max', verbose=1
    )
    ckpt_epoch = ModelCheckpoint(
        filepath=os.path.join(args.model_dir, 'epoch_{epoch:02d}.keras'),
        save_best_only=False, verbose=0
    )
    early_stop = EarlyStopping(
        monitor='val_accuracy', patience=5, mode='max',
        restore_best_weights=True, verbose=1
    )
    csv_logger = CSVLogger(
        filename=os.path.join(args.model_dir, 'training_log.csv'),
        append=(args.mode == 'resume')
    )
    callbacks = [ckpt_best, ckpt_epoch, early_stop, csv_logger]

    # Train the model
    print(f"--- {args.mode.upper()} for {args.epochs} epochs; LR={args.lr}")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=2
    )

    # Save the final model
    final_path = os.path.join(args.model_dir, 'model_final.keras')
    model.save(final_path)
    print(f"Training complete. Final model saved at {final_path}")


if __name__ == "__main__":
    main()
