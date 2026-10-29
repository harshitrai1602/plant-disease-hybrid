"""CPU/CUDA-safe BaselineCNN feature extraction + SVM pipeline.

Run from repository root:
    python scripts/run_hybrid_svm.py

Inputs:
  outputs/baseline_cnn_retrained/best_model.pth
  data/processed/splits/{train,val,test}.csv

Creates a new timestamped folder under outputs/ and never edits the inputs.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ROOT = Path.cwd()
CHECKPOINT = ROOT / "outputs" / "baseline_cnn_retrained" / "best_model.pth"
SPLITS = ROOT / "data" / "processed" / "splits"
OUTPUTS = ROOT / "outputs"
IMAGE_SIZE = 224
BATCH_SIZE = 64  # If memory is tight, change to 32 or 16.
NUM_WORKERS = 0 if os.name == "nt" else 2
SVM_C = 1.0


class BaselineCNN(nn.Module):
    """Architecture must match the checkpoint used by this project."""
    def __init__(self, num_classes: int = 38):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def extract_features(self, x):
        # Shape: (batch_size, 256)
        return torch.flatten(self.features(x), start_dim=1)


def load_checkpoint_file(path: Path):
    # map_location='cpu' allows loading a checkpoint saved on a CUDA machine on CPU.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Older PyTorch versions
        return torch.load(path, map_location="cpu")


def find_state_dict(checkpoint):
    """Verify checkpoint contents and discover the actual state-dict key."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint dict, got {type(checkpoint).__name__}")

    metadata = {
        k: checkpoint[k] for k in (
            "model", "num_classes", "class_names", "best_val_accuracy",
            "best_val_acc", "epoch", "seed"
        ) if k in checkpoint
    }
    candidates = []
    if any(isinstance(k, str) and k.startswith("features.") for k in checkpoint):
        candidates.append(("<top-level state_dict>", checkpoint))

    for key in ("state_dict", "model_state_dict", "model", "net", "weights",
                "best_model_state_dict"):
        if isinstance(checkpoint.get(key), dict):
            candidates.append((key, checkpoint[key]))
    for key, value in checkpoint.items():
        if isinstance(value, dict) and key not in {
            "state_dict", "model_state_dict", "model", "net", "weights",
            "best_model_state_dict"
        }:
            candidates.append((key, value))

    for key_name, candidate in candidates:
        state = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in candidate.items() if isinstance(k, str)
        }
        first = state.get("features.0.weight")
        last = state.get("classifier.2.weight")
        if (first is not None and last is not None
                and tuple(first.shape) == (32, 3, 3, 3)
                and len(last.shape) == 2 and last.shape[1] == 256):
            print(f"Verified state-dict location: {key_name}")
            print(f"Checkpoint classifier class count: {last.shape[0]}")
            print(f"Checkpoint metadata keys: {list(metadata)}")
            return state, metadata

    print("Checkpoint keys:", list(checkpoint.keys()))
    for key, value in checkpoint.items():
        if isinstance(value, dict):
            print(f"Nested dict {key!r} keys (first 15): {list(value)[:15]}")
    raise ValueError(
        "Could not find matching BaselineCNN weights. Inspect the printed keys; "
        "the script intentionally does not guess a state-dict key."
    )


def unique_output_dir():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = OUTPUTS / f"hybrid_svm_run_{stamp}"
    path, suffix = base, 1
    while path.exists():
        path = OUTPUTS / f"{base.name}_{suffix}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def resolve_path(value):
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    for candidate in (ROOT / path, SPLITS / path, ROOT / "data" / path):
        if candidate.exists():
            return candidate.resolve()
    return (ROOT / path).resolve()


def read_split(name):
    path = SPLITS / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required split not found: {path}")
    df = pd.read_csv(path)
    required = {"image_path", "class_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} needs columns {sorted(required)}; found {list(df.columns)}")
    df = df.copy()
    df["class_id"] = pd.to_numeric(df["class_id"], errors="raise").astype(int)
    df["resolved_path"] = df["image_path"].map(resolve_path)
    missing = df.loc[~df["resolved_path"].map(lambda p: p.is_file()), "resolved_path"]
    if len(missing):
        raise FileNotFoundError(
            f"{len(missing)} paths in {name}.csv are missing. Examples:\n" +
            "\n".join(map(str, missing.head(10)))
        )
    return df


class LeafDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = Path(row["resolved_path"])
        try:
            with Image.open(path) as im:
                image = im.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Could not open image: {path}") from exc
        return self.transform(image), int(row["class_id"])


def make_loader(frame, transform, pin_memory):
    return DataLoader(
        LeafDataset(frame, transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )


def extract_features(model, frame, transform, device, split_name, pin_memory):
    loader = make_loader(frame, transform, pin_memory)
    xs, ys = [], []
    started = time.time()
    model.eval()
    with torch.inference_mode():
        for batch_i, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=(device.type == "cuda"))
            features = model.extract_features(images)
            if features.ndim != 2 or features.shape[1] != 256:
                raise RuntimeError(f"Expected 256-D features, got {tuple(features.shape)}")
            xs.append(features.cpu().numpy().astype(np.float32, copy=False))
            ys.append(labels.numpy().astype(np.int64, copy=False))
            if batch_i % 100 == 0 or batch_i == len(loader):
                print(f"  {split_name}: {min(batch_i * BATCH_SIZE, len(frame)):,}/{len(frame):,}")
    X, y = np.concatenate(xs), np.concatenate(ys)
    print(f"{split_name}: feature matrix {X.shape}; elapsed {time.time()-started:.1f}s")
    return X, y


def metrics_for(y, pred, class_ids, class_names):
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_precision": float(precision_score(y, pred, labels=class_ids,
                                                  average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, labels=class_ids,
                                           average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, labels=class_ids,
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "samples": int(len(y)),
    }
    report = classification_report(
        y, pred, labels=class_ids,
        target_names=[class_names.get(i, f"class_{i}") for i in class_ids],
        output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(
        OUT_DIR / f"{CURRENT_SPLIT}_classification_report.csv"
    )
    pd.DataFrame(
        confusion_matrix(y, pred, labels=class_ids),
        index=[class_names.get(i, f"class_{i}") for i in class_ids],
        columns=[class_names.get(i, f"class_{i}") for i in class_ids],
    ).to_csv(OUT_DIR / f"{CURRENT_SPLIT}_confusion_matrix.csv")
    metrics["split"] = CURRENT_SPLIT
    with (OUT_DIR / f"{CURRENT_SPLIT}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"{CURRENT_SPLIT.upper()}: accuracy={metrics['accuracy']:.4%}, "
          f"macro-F1={metrics['macro_f1']:.4%}, weighted-F1={metrics['weighted_f1']:.4%}")
    return metrics


OUT_DIR = None
CURRENT_SPLIT = None


def main():
    global OUT_DIR, CURRENT_SPLIT

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Selected device: {torch.cuda.get_device_name(0) if pin_memory else 'CPU'}")

    train_df, val_df, test_df = (read_split(s) for s in ("train", "val", "test"))
    print(f"Splits: train={len(train_df):,}; val={len(val_df):,}; test={len(test_df):,}")

    all_ids = sorted(set(train_df.class_id) | set(val_df.class_id) | set(test_df.class_id))
    if set(val_df.class_id) - set(train_df.class_id) or set(test_df.class_id) - set(train_df.class_id):
        raise ValueError("Validation/test contain class IDs absent from train split.")
    if all_ids != list(range(len(all_ids))):
        raise ValueError(f"Expected class IDs 0..N-1, found {all_ids}")

    checkpoint = load_checkpoint_file(CHECKPOINT)
    state_dict, metadata = find_state_dict(checkpoint)
    inferred_classes = int(state_dict["classifier.2.weight"].shape[0])
    num_classes = int(metadata.get("num_classes", inferred_classes))
    if num_classes != inferred_classes or num_classes != len(all_ids):
        raise ValueError(
            f"Class-count mismatch: checkpoint metadata={num_classes}, "
            f"checkpoint weights={inferred_classes}, splits={len(all_ids)}"
        )

    model = BaselineCNN(num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    print("Checkpoint loaded with strict=True; CNN weights will not be trained.")

    class_names = {}
    saved_names = metadata.get("class_names")
    if isinstance(saved_names, (list, tuple)) and len(saved_names) == num_classes:
        class_names = {i: str(name) for i, name in enumerate(saved_names)}
    else:
        for frame in (train_df, val_df, test_df):
            if "label" in frame.columns:
                for class_id, group in frame.groupby("class_id"):
                    labels = group["label"].dropna().astype(str).unique()
                    if len(labels) == 1:
                        class_names[int(class_id)] = labels[0]
        for i in all_ids:
            class_names.setdefault(i, f"class_{i}")

    OUT_DIR = unique_output_dir()
    print("Output directory:", OUT_DIR)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    X_train, y_train = extract_features(model, train_df, transform, device, "train", pin_memory)
    X_val, y_val = extract_features(model, val_df, transform, device, "val", pin_memory)
    X_test, y_test = extract_features(model, test_df, transform, device, "test", pin_memory)

    for name, X, y in (
        ("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)
    ):
        np.savez_compressed(OUT_DIR / f"{name}_features.npz", X=X, y=y)

    # LinearSVC scales much better than an RBF SVC for 115k+ training images.
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(C=SVM_C, class_weight="balanced",
                          random_state=42, max_iter=5000)),
    ])
    started = time.time()
    print(f"Fitting LinearSVC on {len(y_train):,} training features.")
    svm.fit(X_train, y_train)
    print(f"SVM fit time: {time.time()-started:.1f}s")
    joblib.dump(svm, OUT_DIR / "hybrid_svm.joblib")

    val_pred = svm.predict(X_val)
    CURRENT_SPLIT = "val"
    val_metrics = metrics_for(y_val, val_pred, all_ids, class_names)

    test_pred = svm.predict(X_test)
    CURRENT_SPLIT = "test"
    test_metrics = metrics_for(y_test, test_pred, all_ids, class_names)

    summary = {
        "pipeline": "BaselineCNN 256-D features + StandardScaler + LinearSVC",
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_metadata": {
            k: (v if isinstance(v, (str, int, float, bool, type(None), list)) else str(v))
            for k, v in metadata.items()
        },
        "image_size": IMAGE_SIZE,
        "feature_dimension": 256,
        "batch_size": BATCH_SIZE,
        "svm": {"type": "LinearSVC", "C": SVM_C, "class_weight": "balanced",
                "max_iter": 5000, "random_state": 42},
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "class_names_by_id": {str(k): v for k, v in class_names.items()},
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "note": "Baseline CNN was not retrained. Existing split CSVs and checkpoint were not modified.",
    }
    with (OUT_DIR / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nCompleted. New artifacts saved to:", OUT_DIR)
    print("Main files: hybrid_svm.joblib, run_summary.json, test_metrics.json, "
          "test_classification_report.csv, test_confusion_matrix.csv")


if __name__ == "__main__":
    main()
