import os
import pandas as pd
import matplotlib.pyplot as plt

# Root folder containing all run subdirectories
MODELS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "models")
MODELS_DIR = os.path.abspath(MODELS_DIR)

if not os.path.isdir(MODELS_DIR):
    raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")

for run_name in sorted(os.listdir(MODELS_DIR)):
    run_dir = os.path.join(MODELS_DIR, run_name)
    log_csv = os.path.join(run_dir, "training_log.csv")
    if not os.path.isfile(log_csv):
        print(f"Skipping {run_name}: no training_log.csv found.")
        continue

    # Load the CSV
    df = pd.read_csv(log_csv)

    # Ensure 'epoch' column exists
    if 'epoch' not in df.columns:
        df['epoch'] = range(1, len(df) + 1)

    # Plot Accuracy
    plt.figure()
    plt.plot(df['epoch'], df['accuracy'], label="train_accuracy")
    plt.plot(df['epoch'], df.get('val_accuracy', []), label="val_accuracy")
    plt.title(f"{run_name} – Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    acc_path = os.path.join(run_dir, "accuracy_plot.png")
    plt.savefig(acc_path)
    plt.close()
    print(f"Saved {acc_path}")

    # Plot Loss
    plt.figure()
    plt.plot(df['epoch'], df['loss'], label="train_loss")
    plt.plot(df['epoch'], df.get('val_loss', []), label="val_loss")
    plt.title(f"{run_name} – Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(run_dir, "loss_plot.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"Saved {loss_path}")
