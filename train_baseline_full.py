from pathlib import Path
import time
import random
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt

from dataset_loader import train_dataset, val_dataset, test_dataset
from baseline_cnn import BaselineCNN


SEED = 42
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 0.001
TRAIN_LIMIT = 115150
VAL_LIMIT = 24675
NUM_CLASSES = 38

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("outputs/baseline_cnn_full")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Baseline CNN training started", flush=True)
print("Device:", DEVICE, flush=True)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# Use a reproducible subset for the initial smoke test.
def make_subset(dataset, limit):
    if limit is None or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)

train_data = make_subset(train_dataset, TRAIN_LIMIT)
val_data = make_subset(val_dataset, VAL_LIMIT)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)
val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)

print("Training samples:", len(train_data), flush=True)
print("Validation samples:", len(val_data), flush=True)
print("Test samples:", len(test_dataset), flush=True)

# Read class names in class_id order.
class_df = pd.read_csv(
    "data/processed/splits/train.csv"
)[["class_id", "label"]].drop_duplicates()

class_df["class_id"] = class_df["class_id"].astype(int)
class_df = class_df.sort_values("class_id")

if class_df["class_id"].tolist() != list(range(NUM_CLASSES)):
    raise ValueError("Expected class IDs 0 through 37.")

class_names = class_df["label"].astype(str).tolist()

model = BaselineCNN(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_accuracy = -1.0
history = []
checkpoint_path = OUTPUT_DIR / "best_model.pth"

# Training and validation
for epoch in range(1, EPOCHS + 1):
    start = time.time()
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for batch_index, (images, labels) in enumerate(train_loader, start=1):
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

        if batch_index % 25 == 0:
            print(
                f"Epoch {epoch}: trained {train_total}/{len(train_data)}",
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
    elapsed = time.time() - start

    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "seconds": elapsed,
    }
    history.append(row)
    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "training_history.csv", index=False
    )

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"train_acc={train_accuracy:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_accuracy:.4f} | "
        f"time={elapsed:.1f}s",
        flush=True,
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_accuracy": best_val_accuracy,
                "num_classes": NUM_CLASSES,
                "class_names": class_names,
                "seed": SEED,
            },
            checkpoint_path,
        )
        print("Saved best checkpoint:", checkpoint_path, flush=True)

# Plot training history.
history_df = pd.DataFrame(history)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train")
axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Validation")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history_df["epoch"], history_df["train_accuracy"], label="Train")
axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="Validation")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
plt.close(fig)


# Test evaluation is intentionally deferred until final model selection.
print("Test evaluation skipped; checkpoint and training curves saved.")
print("Results directory:", OUTPUT_DIR.resolve(), flush=True)
print("Training completed.", flush=True)
