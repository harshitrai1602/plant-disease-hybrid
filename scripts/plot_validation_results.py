
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ANALYSIS = (
    ROOT / "outputs/validation_per_class_analysis_20261101_141250"
)

DEFAULT_BASELINE_CM = (
    ROOT / "outputs/baseline_cnn_retrained/val_confusion_matrix.csv"
)

DEFAULT_HYBRID_CM = (
    ROOT / "outputs/hybrid_svm_run_20261029_125118/val_confusion_matrix.csv"
)


def save_f1_comparison(df, output_dir):
    df = df.sort_values("f1-score_hybrid")

    y = np.arange(len(df))
    height = 0.38

    fig, ax = plt.subplots(figsize=(13, 16))

    ax.barh(
        y - height / 2,
        df["f1-score_baseline"],
        height,
        label="Baseline CNN"
    )

    ax.barh(
        y + height / 2,
        df["f1-score_hybrid"],
        height,
        label="Hybrid CNN + SVM"
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["class_name"], fontsize=8)
    ax.set_xlabel("F1-score")
    ax.set_title("Per-Class F1-Score: Baseline CNN vs Hybrid CNN + SVM")
    ax.set_xlim(0, 1.05)
    ax.legend()
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_dir / "per_class_f1_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


def save_f1_improvement(df, output_dir):
    df = df.sort_values("f1_change")

    fig, ax = plt.subplots(figsize=(12, 15))

    ax.barh(
        df["class_name"],
        df["f1_change"]
    )

    ax.axvline(0, linewidth=1)
    ax.set_xlabel("F1-score change (Hybrid - Baseline)")
    ax.set_title("Per-Class F1-Score Improvement / Regression")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_dir / "per_class_f1_change.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


def save_normalized_confusion_matrix(
    cm_path,
    output_path,
    title,
    expected_classes
):
    cm = pd.read_csv(cm_path, index_col=0)
    cm.index = cm.index.astype(str)
    cm.columns = cm.columns.astype(str)

    if set(cm.index) != set(expected_classes):
        raise ValueError(f"Row labels do not match: {cm_path}")

    if set(cm.columns) != set(expected_classes):
        raise ValueError(f"Column labels do not match: {cm_path}")

    cm = cm.loc[expected_classes, expected_classes]
    values = cm.to_numpy(dtype=float)

    row_totals = values.sum(axis=1, keepdims=True)

    normalized = np.divide(
        values,
        row_totals,
        out=np.zeros_like(values),
        where=row_totals != 0
    )

    fig, ax = plt.subplots(figsize=(19, 16))

    image = ax.imshow(
        normalized,
        interpolation="nearest",
        aspect="auto",
        vmin=0,
        vmax=1
    )

    fig.colorbar(image, ax=ax, label="Proportion of actual class")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")

    ax.set_xticks(np.arange(len(expected_classes)))
    ax.set_yticks(np.arange(len(expected_classes)))

    ax.set_xticklabels(
        expected_classes,
        rotation=90,
        fontsize=6
    )

    ax.set_yticklabels(
        expected_classes,
        fontsize=6
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--analysis-dir",
        default=str(DEFAULT_ANALYSIS)
    )

    parser.add_argument(
        "--baseline-cm",
        default=str(DEFAULT_BASELINE_CM)
    )

    parser.add_argument(
        "--hybrid-cm",
        default=str(DEFAULT_HYBRID_CM)
    )

    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    output_dir = analysis_dir / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = (
        analysis_dir / "baseline_vs_hybrid_per_class.csv"
    )

    if not comparison_path.exists():
        raise FileNotFoundError(comparison_path)

    df = pd.read_csv(comparison_path)

    required_columns = {
        "class_name",
        "f1-score_baseline",
        "f1-score_hybrid",
        "f1_change"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if len(df) != 38:
        raise ValueError(
            f"Expected 38 classes, found {len(df)}"
        )

    df = df.sort_values("class_name").reset_index(drop=True)
    expected_classes = df["class_name"].astype(str).tolist()

    print("=" * 60)
    print("VALIDATION RESULT VISUALIZATION")
    print("=" * 60)

    save_f1_comparison(df, output_dir)
    print("Generated: per_class_f1_comparison.png")

    save_f1_improvement(df, output_dir)
    print("Generated: per_class_f1_change.png")

    save_normalized_confusion_matrix(
        args.baseline_cm,
        output_dir / "baseline_normalized_confusion_matrix.png",
        "Baseline CNN - Normalized Validation Confusion Matrix",
        expected_classes
    )
    print("Generated: baseline_normalized_confusion_matrix.png")

    save_normalized_confusion_matrix(
        args.hybrid_cm,
        output_dir / "hybrid_normalized_confusion_matrix.png",
        "Hybrid CNN + SVM - Normalized Validation Confusion Matrix",
        expected_classes
    )
    print("Generated: hybrid_normalized_confusion_matrix.png")

    print("\nAll graphs saved successfully.")
    print("Output directory:", output_dir.resolve())
    print("=" * 60)


if __name__ == "__main__":
    main()
