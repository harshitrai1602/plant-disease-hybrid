from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Run this script from the project root:
# python scripts/compare_test_results.py

BASELINE_DIR = Path("outputs/baseline_cnn_retrained")
HYBRID_DIR = Path("outputs/hybrid_svm_run_20261029_125118")

BASELINE_REPORT = BASELINE_DIR / "test_classification_report.csv"
HYBRID_REPORT = HYBRID_DIR / "test_classification_report.csv"
BASELINE_CM = BASELINE_DIR / "test_confusion_matrix.csv"
HYBRID_CM = HYBRID_DIR / "test_confusion_matrix.csv"
BASELINE_METRICS = BASELINE_DIR / "test_metrics.json"
HYBRID_METRICS = HYBRID_DIR / "test_metrics.json"

AGGREGATE_ROWS = {"accuracy", "macro avg", "weighted avg", "samples avg"}


def load_report(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing classification report: {path}")
    report = pd.read_csv(path, index_col=0)
    report.index = report.index.astype(str).str.strip()
    report = report.loc[~report.index.isin(AGGREGATE_ROWS)].copy()
    required = {"precision", "recall", "f1-score", "support"}
    missing = required.difference(report.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return report.sort_index()


def load_confusion_matrix(path: Path, labels: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing confusion matrix: {path}")
    cm = pd.read_csv(path, index_col=0)
    cm.index = cm.index.astype(str).str.strip()
    cm.columns = cm.columns.astype(str).str.strip()
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    return cm.astype(float)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics JSON: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_rows(cm: pd.DataFrame) -> pd.DataFrame:
    row_sums = cm.sum(axis=1).replace(0, np.nan)
    return cm.div(row_sums, axis=0).fillna(0)


def save_confusion_plot(cm: pd.DataFrame, title: str, output: Path) -> None:
    normalized = normalize_rows(cm).to_numpy()
    labels = cm.index.tolist()

    fig, ax = plt.subplots(figsize=(17, 15))
    image = ax.imshow(normalized, interpolation="nearest", aspect="auto", vmin=0, vmax=1)
    ax.set_title(title, fontsize=16, pad=16)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized proportion")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for required_path in [
        BASELINE_REPORT, HYBRID_REPORT, BASELINE_CM, HYBRID_CM,
        BASELINE_METRICS, HYBRID_METRICS
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {required_path}\n"
                "Check that you are running this script from the project root."
            )

    baseline = load_report(BASELINE_REPORT)
    hybrid = load_report(HYBRID_REPORT)

    baseline_labels = set(baseline.index)
    hybrid_labels = set(hybrid.index)
    if baseline_labels != hybrid_labels:
        raise ValueError(
            "Class names differ between reports.\n"
            f"Only baseline: {sorted(baseline_labels - hybrid_labels)}\n"
            f"Only hybrid: {sorted(hybrid_labels - baseline_labels)}"
        )

    labels = sorted(baseline.index.tolist())
    baseline = baseline.reindex(labels)
    hybrid = hybrid.reindex(labels)

    if not np.allclose(
        baseline["support"].to_numpy(),
        hybrid["support"].to_numpy()
    ):
        raise ValueError("Class support counts differ between baseline and hybrid.")

    baseline_cm = load_confusion_matrix(BASELINE_CM, labels)
    hybrid_cm = load_confusion_matrix(HYBRID_CM, labels)

    expected_samples = int(baseline["support"].sum())
    if int(hybrid["support"].sum()) != expected_samples:
        raise ValueError("Total support differs between baseline and hybrid.")
    if int(baseline_cm.to_numpy().sum()) != expected_samples:
        raise ValueError("Baseline confusion matrix total does not match report support.")
    if int(hybrid_cm.to_numpy().sum()) != expected_samples:
        raise ValueError("Hybrid confusion matrix total does not match report support.")

    baseline_json = load_json(BASELINE_METRICS)
    hybrid_json = load_json(HYBRID_METRICS)

    comparison = pd.DataFrame({
        "class": labels,
        "support": baseline["support"].to_numpy(),
        "baseline_precision": baseline["precision"].to_numpy(),
        "hybrid_precision": hybrid["precision"].to_numpy(),
        "baseline_recall": baseline["recall"].to_numpy(),
        "hybrid_recall": hybrid["recall"].to_numpy(),
        "baseline_f1": baseline["f1-score"].to_numpy(),
        "hybrid_f1": hybrid["f1-score"].to_numpy(),
    })
    comparison["f1_change"] = comparison["hybrid_f1"] - comparison["baseline_f1"]
    comparison["f1_change_percentage_points"] = comparison["f1_change"] * 100
    comparison = comparison.sort_values("f1_change", ascending=False)

    baseline_accuracy = float(baseline_json.get("test_accuracy"))
    hybrid_accuracy = float(hybrid_json.get("accuracy"))
    baseline_macro_f1 = float(baseline["f1-score"].mean())
    hybrid_macro_f1 = float(hybrid["f1-score"].mean())
    baseline_weighted_f1 = float(
        np.average(baseline["f1-score"], weights=baseline["support"])
    )
    hybrid_weighted_f1 = float(
        np.average(hybrid["f1-score"], weights=hybrid["support"])
    )

    metrics = pd.DataFrame([
        {"metric": "Accuracy", "baseline": baseline_accuracy, "hybrid": hybrid_accuracy},
        {"metric": "Macro precision", "baseline": baseline["precision"].mean(),
         "hybrid": hybrid["precision"].mean()},
        {"metric": "Macro recall", "baseline": baseline["recall"].mean(),
         "hybrid": hybrid["recall"].mean()},
        {"metric": "Macro F1", "baseline": baseline_macro_f1,
         "hybrid": hybrid_macro_f1},
        {"metric": "Weighted F1", "baseline": baseline_weighted_f1,
         "hybrid": hybrid_weighted_f1},
    ])
    metrics["improvement_percentage_points"] = (
        metrics["hybrid"] - metrics["baseline"]
    ) * 100

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / f"test_evaluation_comparison_{timestamp}"
    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(output_dir / "test_per_class_comparison.csv", index=False)
    metrics.to_csv(output_dir / "test_metrics_comparison.csv", index=False)
    baseline_cm.to_csv(output_dir / "baseline_test_confusion_matrix.csv")
    hybrid_cm.to_csv(output_dir / "hybrid_test_confusion_matrix.csv")
    normalize_rows(baseline_cm).to_csv(
        output_dir / "baseline_test_confusion_matrix_normalized.csv"
    )
    normalize_rows(hybrid_cm).to_csv(
        output_dir / "hybrid_test_confusion_matrix_normalized.csv"
    )

    # Overall metric comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.36
    ax.bar(x - width / 2, metrics["baseline"] * 100, width, label="Baseline CNN")
    ax.bar(x + width / 2, metrics["hybrid"] * 100, width, label="Hybrid CNN + SVM")
    ax.set_ylabel("Score (%)")
    ax.set_title("Test-set metric comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["metric"], rotation=15, ha="right")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(graph_dir / "test_metrics_comparison.png", dpi=300)
    plt.close(fig)

    # Per-class F1 comparison
    plot_data = comparison.sort_values("baseline_f1")
    y = np.arange(len(plot_data))
    fig, ax = plt.subplots(figsize=(13, 15))
    ax.barh(y - 0.18, plot_data["baseline_f1"], height=0.35, label="Baseline CNN")
    ax.barh(y + 0.18, plot_data["hybrid_f1"], height=0.35, label="Hybrid CNN + SVM")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_data["class"], fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1-score")
    ax.set_title("Per-class test F1-score comparison")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(graph_dir / "test_per_class_f1_comparison.png", dpi=300)
    plt.close(fig)

    # Per-class F1 change
    change_data = comparison.sort_values("f1_change")
    fig, ax = plt.subplots(figsize=(13, 15))
    ax.barh(change_data["class"], change_data["f1_change"] * 100)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Hybrid minus baseline F1 (percentage points)")
    ax.set_title("Per-class test F1-score change")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(graph_dir / "test_per_class_f1_change.png", dpi=300)
    plt.close(fig)

    save_confusion_plot(
        baseline_cm, "Baseline CNN: row-normalized test confusion matrix",
        graph_dir / "baseline_test_normalized_confusion_matrix.png"
    )
    save_confusion_plot(
        hybrid_cm, "Hybrid CNN + SVM: row-normalized test confusion matrix",
        graph_dir / "hybrid_test_normalized_confusion_matrix.png"
    )

    summary = {
        "test_samples": expected_samples,
        "num_classes": len(labels),
        "class_names_match": True,
        "class_supports_match": True,
        "baseline_accuracy": baseline_accuracy,
        "hybrid_accuracy": hybrid_accuracy,
        "accuracy_improvement_percentage_points": (hybrid_accuracy - baseline_accuracy) * 100,
        "baseline_macro_f1": baseline_macro_f1,
        "hybrid_macro_f1": hybrid_macro_f1,
        "macro_f1_improvement_percentage_points": (hybrid_macro_f1 - baseline_macro_f1) * 100,
        "output_directory": str(output_dir),
    }
    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTEST COMPARISON COMPLETED")
    print("------------------------")
    print(f"Test samples: {expected_samples}")
    print(f"Classes: {len(labels)}")
    print(f"Baseline accuracy: {baseline_accuracy:.4%}")
    print(f"Hybrid accuracy:   {hybrid_accuracy:.4%}")
    print(f"Accuracy gain:     {(hybrid_accuracy - baseline_accuracy) * 100:.2f} percentage points")
    print(f"Baseline macro F1: {baseline_macro_f1:.4%}")
    print(f"Hybrid macro F1:   {hybrid_macro_f1:.4%}")
    print(f"Macro F1 gain:     {(hybrid_macro_f1 - baseline_macro_f1) * 100:.2f} percentage points")
    print(f"\nSaved all files to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
