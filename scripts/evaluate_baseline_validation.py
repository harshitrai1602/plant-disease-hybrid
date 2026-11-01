
from pathlib import Path
import sys
import json
import os

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_cnn import BaselineCNN

VAL_CSV = ROOT / "data" / "processed" / "splits" / "val.csv"

CHECKPOINT = (
    ROOT / "outputs" / "baseline_cnn_retrained" / "best_model.pth"
)

OUTPUT_DIR = ROOT / "outputs" / "baseline_cnn_retrained"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
NUM_CLASSES = 38

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# --------------------------------------------------
# 2. VALIDATION TRANSFORM
# Same as dataset_loader.py
# --------------------------------------------------

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# 3. RESOLVE IMAGE PATHS
# --------------------------------------------------

def resolve_image_path(raw_path):

    raw_path = str(raw_path).strip().strip('"')
    original = Path(raw_path)

    if original.is_file():
        return original

    normalized = raw_path.replace("/", "\\")
    marker = "plant-disease-hybrid\\"

    if marker.lower() in normalized.lower():

        index = normalized.lower().rfind(
            marker.lower()
        )

        relative_part = normalized[
            index + len(marker):
        ]

        candidate = ROOT / Path(relative_part)

        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Validation image not found: {raw_path}"
    )

# --------------------------------------------------
# 4. VALIDATION DATASET
# --------------------------------------------------

class ValidationDataset(Dataset):

    def __init__(self, csv_path, class_names):

        self.data = pd.read_csv(csv_path)

        required = {"image_path", "label", "class_id"}

        if not required.issubset(self.data.columns):
            raise ValueError(
                f"Missing columns: {required - set(self.data.columns)}"
            )

        self.paths = [
            resolve_image_path(p)
            for p in self.data["image_path"]
        ]

        self.labels = self.data["class_id"].astype(int).tolist()

        self.class_names = class_names

        self.transform = eval_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        with Image.open(self.paths[index]) as image:

            image = image.convert("RGB")
            image = self.transform(image)

        label = self.labels[index]

        return image, label

# --------------------------------------------------
# 5. LOAD EXISTING CHECKPOINT
# --------------------------------------------------

def load_checkpoint(path):

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

    except TypeError:

        checkpoint = torch.load(
            path,
            map_location="cpu"
        )

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a dictionary.")

    candidates = []

    if any(
        isinstance(k, str) and k.startswith("features.")
        for k in checkpoint
    ):
        candidates.append(checkpoint)

    possible_keys = [
        "state_dict",
        "model_state_dict",
        "model",
        "net",
        "weights",
        "best_model_state_dict"
    ]

    for key in possible_keys:

        value = checkpoint.get(key)

        if isinstance(value, dict):
            candidates.append(value)

    for key, value in checkpoint.items():

        if isinstance(value, dict) and key not in possible_keys:
            candidates.append(value)

    for candidate in candidates:

        state = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in candidate.items()
            if isinstance(k, str)
        }

        first = state.get("features.0.weight")
        last = state.get("classifier.2.weight")

        if (
            first is not None
            and last is not None
            and tuple(first.shape) == (32, 3, 3, 3)
            and len(last.shape) == 2
            and last.shape[1] == 256
        ):

            if last.shape[0] != NUM_CLASSES:
                raise ValueError(
                    f"Checkpoint has {last.shape[0]} classes, "
                    f"expected {NUM_CLASSES}."
                )

            return state

    raise ValueError(
        "Could not identify the BaselineCNN state dictionary. "
        "Checkpoint has not been modified."
    )

# --------------------------------------------------
# 6. MAIN VALIDATION INFERENCE
# --------------------------------------------------

def main():

    print("=" * 60)
    print("BASELINE CNN - VALIDATION ONLY EVALUATION")
    print("=" * 60)

    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    if not VAL_CSV.is_file():
        raise FileNotFoundError(VAL_CSV)

    # Read validation metadata
    val_df = pd.read_csv(VAL_CSV)

    if len(val_df) != 24675:
        raise ValueError(
            f"Expected 24675 validation rows, found {len(val_df)}"
        )

    # Build class ID to class name mapping
    class_mapping = (
        val_df.groupby("class_id")["label"]
        .agg(lambda s: s.mode().iloc[0])
        .to_dict()
    )

    if set(class_mapping.keys()) != set(range(NUM_CLASSES)):
        raise ValueError(
            "Validation class IDs are not exactly 0-37."
        )

    class_names = [
        str(class_mapping[i])
        for i in range(NUM_CLASSES)
    ]

    if len(set(class_names)) != NUM_CLASSES:
        raise ValueError("Duplicate class names detected.")

    print("Validation images:", len(val_df))
    print("Number of classes:", len(class_names))

    # Create dataset and loader
    dataset = ValidationDataset(
        VAL_CSV,
        class_names
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    # Load checkpoint without training
    model = BaselineCNN(
        num_classes=NUM_CLASSES
    )

    state_dict = load_checkpoint(CHECKPOINT)

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(DEVICE)
    model.eval()

    print("\nCheckpoint loaded successfully.")
    print("Model set to evaluation mode.")

    # --------------------------------------------------
    # 7. PREDICTION
    # --------------------------------------------------

    y_true = []
    y_pred = []

    print("\nRunning validation inference...")

    with torch.inference_mode():

        for batch_index, (images, labels) in enumerate(loader):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            y_true.extend(labels.numpy().tolist())

            y_pred.extend(
                predictions.cpu().numpy().tolist()
            )

            if (batch_index + 1) % 100 == 0:

                print(
                    f"Processed batches: "
                    f"{batch_index + 1}/{len(loader)}"
                )

    # Verify all images were evaluated
    if len(y_true) != len(dataset):
        raise RuntimeError(
            "Prediction count does not match validation dataset."
        )

    print("\nInference completed.")

    # --------------------------------------------------
    # 8. CALCULATE METRICS
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    # --------------------------------------------------
    # 9. PREVENT OVERWRITING EXISTING RESULTS
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = OUTPUT_DIR / "val_classification_report.csv"
    cm_path = OUTPUT_DIR / "val_confusion_matrix.csv"
    metrics_path = OUTPUT_DIR / "val_metrics.json"

    output_files = [
        report_path,
        cm_path,
        metrics_path
    ]

    existing = [
        str(path)
        for path in output_files
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Output files already exist. Nothing was overwritten:\n"
            + "\n".join(existing)
        )

    # --------------------------------------------------
    # 10. SAVE CLASSIFICATION REPORT
    # --------------------------------------------------

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        report_path,
        index=True,
        index_label="class_name"
    )

    # --------------------------------------------------
    # 11. SAVE CONFUSION MATRIX
    # --------------------------------------------------

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )

    cm_df.index.name = "true_class"
    cm_df.columns.name = "predicted_class"

    cm_df.to_csv(cm_path)

    # --------------------------------------------------
    # 12. SAVE METRICS SUMMARY
    # --------------------------------------------------

    metrics = {
        "model": "BaselineCNN",
        "evaluation_type": "validation_only",
        "dataset": "data/processed/splits/val.csv",
        "total_validation_images": len(y_true),
        "num_classes": NUM_CLASSES,
        "device": str(DEVICE),
        "accuracy": float(accuracy),
        "macro_precision": float(
            report["macro avg"]["precision"]
        ),
        "macro_recall": float(
            report["macro avg"]["recall"]
        ),
        "macro_f1": float(
            report["macro avg"]["f1-score"]
        ),
        "weighted_f1": float(
            report["weighted avg"]["f1-score"]
        ),
        "checkpoint": str(CHECKPOINT),
        "training_performed": False,
        "model_fitting_performed": False,
        "test_dataset_loaded": False
    }

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # --------------------------------------------------
    # 13. FINAL SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("BASELINE CNN VALIDATION RESULTS")
    print("=" * 60)

    print(f"Validation images : {len(y_true)}")
    print(f"Classes           : {NUM_CLASSES}")
    print(f"Accuracy          : {accuracy:.4f}")
    print(
        f"Macro Precision   : "
        f"{metrics['macro_precision']:.4f}"
    )
    print(
        f"Macro Recall      : "
        f"{metrics['macro_recall']:.4f}"
    )
    print(
        f"Macro F1          : "
        f"{metrics['macro_f1']:.4f}"
    )
    print(
        f"Weighted F1       : "
        f"{metrics['weighted_f1']:.4f}"
    )

    print("\nFiles generated successfully:")

    print(report_path)
    print(cm_path)
    print(metrics_path)

    print("\nNo model training or fitting performed.")
    print("No test dataset loaded.")
    print("Existing checkpoints and test artifacts preserved.")

    print("=" * 60)


if __name__ == "__main__":
    main()
