import subprocess
import datetime
import os

# location of the training entrypoint
TRAIN_SCRIPT = os.path.join("train.py")

# arguments shared across every experiment
COMMON_ARGS = [
    "--mode", "train",
    "--features", "../data/features",
    "--batch_size", "32",
    "--epochs", "30"
]

# different experiment configurations
EXPERIMENTS = [
    {"name": "run_base", "model_dir": "../models/run_base", "batch_size": "16", "epochs": "15"},
    {"name": "run_relu", "model_dir": "../models/run_relu", "batch_size": "16", "epochs": "15", "activation": "relu"},
    {"name": "run_tanh", "model_dir": "../models/run_tanh", "batch_size": "16", "epochs": "15", "activation": "tanh"},
    {"name": "run_elu", "model_dir": "../models/run_elu", "batch_size": "16", "epochs": "15", "activation": "elu"},
    {"name": "run_selu", "model_dir": "../models/run_selu", "batch_size": "16", "epochs": "15", "activation": "selu"},
    {"name": "run_gelu", "model_dir": "../models/run_gelu", "batch_size": "16", "epochs": "15", "activation": "gelu"},
    {"name": "run_bs8", "model_dir": "../models/run_bs8", "batch_size": "8", "epochs": "15", "activation": "relu"},
    {"name": "run_bs32", "model_dir": "../models/run_bs32", "batch_size": "32", "epochs": "15", "activation": "relu"},
    {"name": "run_smallFilt", "model_dir": "../models/run_smallFilt", "batch_size": "16", "epochs": "15",
     "filters": ["16", "32", "64"]},
    {"name": "run_bigFilt", "model_dir": "../models/run_bigFilt", "batch_size": "16", "epochs": "15",
     "filters": ["64", "128", "256"]},
    {"name": "run_dropHigh", "model_dir": "../models/run_dropHigh", "batch_size": "16", "epochs": "15",
     "dropouts": ["0.3", "0.3", "0.3", "0.6"]},
]

LOG_FILE = "overnight.log"


def build_cmd(config):
    cmd = ["python", TRAIN_SCRIPT] + COMMON_ARGS
    # override batch size and epoch count
    if "batch_size" in config:
        i = cmd.index("--batch_size") + 1
        cmd[i] = config["batch_size"]
    if "epochs" in config:
        i = cmd.index("--epochs") + 1
        cmd[i] = config["epochs"]

    # always specify output folder
    cmd += ["--model_dir", config["model_dir"]]

    # add optional settings
    if "lr" in config:
        cmd += ["--lr", config["lr"]]
    if "activation" in config:
        cmd += ["--activation", config["activation"]]
    if "filters" in config:
        cmd += ["--filters"] + config["filters"]
    if "dropouts" in config:
        cmd += ["--dropouts"] + config["dropouts"]

    return cmd


def main():
    # append log entries to file
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n=== BATCH START @ {datetime.datetime.now()} ===\n")
        for cfg in EXPERIMENTS:
            log.write(f"\n--- RUN {cfg['name']} @ {datetime.datetime.now()} ---\n")
            command = build_cmd(cfg)
            log.write("COMMAND: " + " ".join(command) + "\n")
            log.flush()

            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore"
            )

            # stream the training output
            for line in proc.stdout:
                print(line, end="")
                log.write(line)
            proc.wait()

            log.write(f"EXIT CODE: {proc.returncode}\n")
            log.flush()

        log.write(f"\n=== BATCH END @ {datetime.datetime.now()} ===\n")

    print(f"\nAll experiments done — see {LOG_FILE} for details.")


if __name__ == "__main__":
    main()
