
from pathlib import Path
import json

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from dataset_loader import test_dataset
from baseline_cnn import BaselineCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = Path("outputs/baseline_cnn_retrained/best_model.pth")
OUTPUT_DIR = Path("outputs/baseline_cnn_retrained")
BATCH_SIZE = 16
NUM_WORKERS = 2


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    model = BaselineCNN(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    all_labels = []
    all_predictions = []

    print("Device:", DEVICE)
    print("Test images:", len(test_dataset))
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Best validation accuracy:",
          checkpoint["best_val_accuracy"])

    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images = images.to(DEVICE)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

            if batch_idx % 100 == 0:
                print(
                    f"Evaluated {len(all_labels)}/{len(test_dataset)} images",
                    flush=True,
                )

    accuracy = accuracy_score(all_labels, all_predictions)

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(OUTPUT_DIR / "test_classification_report.csv")

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(num_classes)),
    )
    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    ).to_csv(OUTPUT_DIR / "test_confusion_matrix.csv")

    results = {
        "checkpoint_epoch": checkpoint["epoch"],
        "best_validation_accuracy": checkpoint["best_val_accuracy"],
        "test_accuracy": accuracy,
        "test_samples": len(all_labels),
        "num_classes": num_classes,
    }

    with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nTEST RESULTS")
    print("=" * 40)
    print(f"Test accuracy: {accuracy:.4%}")
    print(f"Test samples: {len(all_labels)}")
    print("Classification report saved.")
    print("Confusion matrix saved.")
    print("Results directory:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
