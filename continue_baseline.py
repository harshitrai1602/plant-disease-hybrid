
from pathlib import Path
import os
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from dataset_loader import train_dataset, val_dataset, test_dataset
from baseline_cnn import BaselineCNN


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
BATCH_SIZE = 16
EPOCHS = 20                 # Total target epochs for this new run
LEARNING_RATE = 0.001
TRAIN_LIMIT = 115150
VAL_LIMIT = 24675
NUM_CLASSES = 38
NUM_WORKERS = 4

# New directory: historical results remain untouched.
OUTPUT_DIR = Path("outputs/baseline_cnn_retrained")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAST_CHECKPOINT = OUTPUT_DIR / "last_checkpoint.pth"
BEST_CHECKPOINT = OUTPUT_DIR / "best_model.pth"
HISTORY_FILE = OUTPUT_DIR / "training_history.csv"
CURVES_FILE = OUTPUT_DIR / "training_curves.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("Fresh/resumable Baseline CNN training", flush=True)
    print("Device:", DEVICE, flush=True)
    print("Batch size:", BATCH_SIZE, flush=True)
    print("DataLoader workers:", NUM_WORKERS, flush=True)

    # Limit CPU thread oversubscription. Benchmark this setting
    # against other values if CPU performance is poor.
    
    if DEVICE.type == "cpu":
     torch.set_num_threads(8)
     print("PyTorch CPU threads:", torch.get_num_threads(), flush=True)

    else:
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    def make_subset(dataset, limit):
        if limit is None or limit >= len(dataset):
            return dataset
        generator = torch.Generator().manual_seed(SEED)
        indices = torch.randperm(
            len(dataset), generator=generator
        )[:limit].tolist()
        return Subset(dataset, indices)

    train_data = make_subset(train_dataset, TRAIN_LIMIT)
    val_data = make_subset(val_dataset, VAL_LIMIT)

    train_generator = torch.Generator().manual_seed(SEED)

    loader_options = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
    }

    if NUM_WORKERS > 0:
        loader_options["persistent_workers"] = True
        loader_options["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_data,
        shuffle=True,
        generator=train_generator,
        **loader_options,
    )

    val_loader = DataLoader(
        val_data,
        shuffle=False,
        **loader_options,
    )

    print("Training samples:", len(train_data), flush=True)
    print("Validation samples:", len(val_data), flush=True)
    print("Test samples:", len(test_dataset), flush=True)

    # --------------------------------------------------------
    # CLASS NAMES
    # --------------------------------------------------------

    class_df = pd.read_csv(
        "data/processed/splits/train.csv"
    )[["class_id", "label"]].drop_duplicates()

    class_df["class_id"] = class_df["class_id"].astype(int)
    class_df = class_df.sort_values("class_id")

    if class_df["class_id"].tolist() != list(range(NUM_CLASSES)):
        raise ValueError("Expected class IDs 0 through 37.")

    class_names = class_df["label"].astype(str).tolist()

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = BaselineCNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    start_epoch = 1
    best_val_accuracy = -1.0
    history = []

    # --------------------------------------------------------
    # RESUME, IF A CHECKPOINT EXISTS
    # --------------------------------------------------------

    if LAST_CHECKPOINT.exists():
        print("Restoring latest checkpoint...", flush=True)

        checkpoint = torch.load(
            LAST_CHECKPOINT,
            map_location=DEVICE,
            weights_only=False,
        )

        if checkpoint["num_classes"] != NUM_CLASSES:
            raise ValueError("Checkpoint class count mismatch.")

        if checkpoint["class_names"] != class_names:
            raise ValueError("Checkpoint class names mismatch.")

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_accuracy = checkpoint["best_val_accuracy"]
        history = checkpoint.get("history", [])

        train_generator.set_state(
            checkpoint["train_generator_state"].cpu()
        )

        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        np.random.set_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["python_rng_state"])

        if DEVICE.type == "cuda" and "cuda_rng_states" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])

        print("Resuming at epoch:", start_epoch, flush=True)

    else:
        print("No checkpoint found; starting from epoch 1.", flush=True)

    # --------------------------------------------------------
    # CHECKPOINT WRITER
    # --------------------------------------------------------

    def save_checkpoint(path, epoch):
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "num_classes": NUM_CLASSES,
            "class_names": class_names,
            "seed": SEED,
            "history": history,
            "train_generator_state": train_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }

        if DEVICE.type == "cuda":
            state["cuda_rng_states"] = torch.cuda.get_rng_state_all()

        temp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temp_path)
        os.replace(temp_path, path)

    # --------------------------------------------------------
    # TRAIN + VALIDATE
    # --------------------------------------------------------

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()

        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(
            train_loader, start=1
        ):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            count = labels.size(0)
            train_loss_sum += loss.item() * count
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += count

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch}/{EPOCHS}: "
                    f"{train_total}/{len(train_data)} samples",
                    flush=True,
                )

        train_loss = train_loss_sum / train_total
        train_accuracy = train_correct / train_total

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                count = labels.size(0)
                val_loss_sum += loss.item() * count
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += count

        val_loss = val_loss_sum / val_total
        val_accuracy = val_correct / val_total
        elapsed = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "seconds": elapsed,
        })

        # Write history atomically, preserving earlier epochs
        # from THIS new run.
        history_df = pd.DataFrame(history)
        temp_csv = HISTORY_FILE.with_suffix(".csv.tmp")
        history_df.to_csv(temp_csv, index=False)
        os.replace(temp_csv, HISTORY_FILE)

        improved = val_accuracy > best_val_accuracy
        if improved:
            best_val_accuracy = val_accuracy

        # Save latest checkpoint every completed epoch.
        save_checkpoint(LAST_CHECKPOINT, epoch)

        if improved:
            save_checkpoint(BEST_CHECKPOINT, epoch)
            print("Updated best checkpoint.", flush=True)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.4f} | "
            f"time={elapsed / 60:.1f} min",
            flush=True,
        )

        # Refresh the curves after each epoch.
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(
            history_df["epoch"], history_df["train_loss"], label="Train"
        )
        axes[0].plot(
            history_df["epoch"], history_df["val_loss"], label="Validation"
        )
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(
            history_df["epoch"],
            history_df["train_accuracy"],
            label="Train",
        )
        axes[1].plot(
            history_df["epoch"],
            history_df["val_accuracy"],
            label="Validation",
        )
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(CURVES_FILE, dpi=150)
        plt.close(fig)

    print("\nTraining session completed.", flush=True)
    print("Latest checkpoint:", LAST_CHECKPOINT.resolve(), flush=True)
    print("Best checkpoint:", BEST_CHECKPOINT.resolve(), flush=True)
    print("History:", HISTORY_FILE.resolve(), flush=True)
    print("Test evaluation is deferred.", flush=True)


if __name__ == "__main__":
    main()
