
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

HYBRID_RUN = (
    PROJECT_ROOT
    / "outputs"
    / "hybrid_svm_run_20261029_125118"
)

CLASS_PAIRS = [
    ("Corn_Gray_leaf_spot", "Corn_Northern_Leaf_Blight"),
    ("Tomato_Early_blight", "Tomato_Late_blight"),
]

REPORT_NAMES = [
    "val_classification_report.csv",
    "validation_classification_report.csv",
    "val_report.csv",
]

CM_NAMES = [
    "val_confusion_matrix.csv",
    "validation_confusion_matrix.csv",
    "val_cm.csv",
]


def find_file(root: Path, names: list[str], preferred: Path | None = None):
    """Find a saved validation artifact without modifying it."""
    if preferred:
        if not preferred.is_file():
            raise FileNotFoundError(f"File not found: {preferred}")
        return preferred

    if not root.exists():
        return None

    matches = []
    for name in names:
        matches.extend(root.rglob(name))

    matches = sorted(set(matches))
    # Never mistake hybrid reports for baseline reports.
    if root.name == "outputs":
     matches = [
        p for p in matches
        if "baseline" in str(p).lower()
    ]

    if not matches:
        return None

    # Prefer artifacts inside the baseline CNN output directory.
    baseline_matches = [
        p for p in matches
        if "baseline" in str(p).lower()
    ]

    return baseline_matches[0] if baseline_matches else matches[0]


def load_classification_report(path: Path) -> pd.DataFrame:
    """Load a sklearn-style classification report CSV."""
    df = pd.read_csv(path)

    # Handle CSVs saved with the class names as the index.
    possible_index_columns = [
        col for col in df.columns
        if str(col).lower().startswith("unnamed")
        or str(col).lower() in {"class", "label", "class_name"}
    ]

    if possible_index_columns:
        first = possible_index_columns[0]
        df = df.rename(columns={first: "class_name"})
    elif "class_name" not in df.columns:
        df = df.rename(columns={df.columns[0]: "class_name"})

    df["class_name"] = df["class_name"].astype(str).str.strip()

    # Remove aggregate rows; retain only individual classes.
    aggregate_names = {
        "accuracy",
        "macro avg",
        "weighted avg",
        "micro avg",
        "samples avg",
    }

    df = df[
        ~df["class_name"].str.lower().isin(aggregate_names)
    ].copy()

    required = {"precision", "recall", "f1-score", "support"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["precision", "recall", "f1-score"])
    return df[["class_name", "precision", "recall", "f1-score", "support"]]


def load_confusion_matrix(path: Path) -> pd.DataFrame:
    """
    Load a labeled confusion matrix.
    Expected layout: rows = true classes, columns = predicted classes.
    """
    df = pd.read_csv(path, index_col=0)

    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()

    # Remove accidental unnamed columns.
    df = df.loc[
        :,
        [c for c in df.columns if not c.lower().startswith("unnamed")]
    ]

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


def compare_reports(
    baseline: pd.DataFrame,
    hybrid: pd.DataFrame,
) -> pd.DataFrame:
    merged = baseline.merge(
        hybrid,
        on="class_name",
        how="outer",
        suffixes=("_baseline", "_hybrid"),
        indicator=True,
    )

    if not (merged["_merge"] == "both").all():
        missing = merged.loc[
            merged["_merge"] != "both",
            ["class_name", "_merge"]
        ]
        raise ValueError(
            "Baseline and hybrid reports do not contain the same classes:\n"
            + missing.to_string(index=False)
        )

    merged["precision_change"] = (
        merged["precision_hybrid"] - merged["precision_baseline"]
    )
    merged["recall_change"] = (
        merged["recall_hybrid"] - merged["recall_baseline"]
    )
    merged["f1_change"] = (
        merged["f1-score_hybrid"] - merged["f1-score_baseline"]
    )

    merged["f1_change_percentage_points"] = (
        merged["f1_change"] * 100
    )

    return merged.drop(columns="_merge").sort_values(
        "f1_change", ascending=True
    )


def pair_error_analysis(
    matrix: pd.DataFrame,
    model_name: str,
) -> list[dict]:
    results = []

    for class_a, class_b in CLASS_PAIRS:
        missing = [
            name for name in (class_a, class_b)
            if name not in matrix.index or name not in matrix.columns
        ]

        if missing:
            results.append({
                "model": model_name,
                "class_a": class_a,
                "class_b": class_b,
                "status": "Missing labels in confusion matrix: "
                          + ", ".join(missing),
            })
            continue

        a_to_b = int(matrix.loc[class_a, class_b])
        b_to_a = int(matrix.loc[class_b, class_a])

        a_correct = int(matrix.loc[class_a, class_a])
        b_correct = int(matrix.loc[class_b, class_b])

        a_total = int(matrix.loc[class_a].sum())
        b_total = int(matrix.loc[class_b].sum())

        results.append({
            "model": model_name,
            "class_a": class_a,
            "class_b": class_b,
            "a_true_predicted_as_b": a_to_b,
            "b_true_predicted_as_a": b_to_a,
            "combined_pair_confusions": a_to_b + b_to_a,
            "a_correct": a_correct,
            "a_total": a_total,
            "a_recall_from_matrix": (
                a_correct / a_total if a_total else None
            ),
            "b_correct": b_correct,
            "b_total": b_total,
            "b_recall_from_matrix": (
                b_correct / b_total if b_total else None
            ),
            "status": "OK",
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare saved baseline and hybrid validation results."
    )

    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=None,
        help="Path to baseline validation classification report CSV.",
    )
    parser.add_argument(
        "--baseline-cm",
        type=Path,
        default=None,
        help="Path to baseline validation confusion matrix CSV.",
    )
    parser.add_argument(
        "--hybrid-report",
        type=Path,
        default=None,
        help="Path to hybrid validation classification report CSV.",
    )
    parser.add_argument(
        "--hybrid-cm",
        type=Path,
        default=None,
        help="Path to hybrid validation confusion matrix CSV.",
    )

    args = parser.parse_args()

    # Auto-discover baseline artifacts only under baseline output folders.
    baseline_root = PROJECT_ROOT / "outputs" / "baseline_cnn"

    if not baseline_root.exists():
        baseline_root = PROJECT_ROOT / "outputs"

    baseline_report = find_file(
        baseline_root, REPORT_NAMES, args.baseline_report
    )
    baseline_cm = find_file(
        baseline_root, CM_NAMES, args.baseline_cm
    )

    hybrid_report = find_file(
        HYBRID_RUN, REPORT_NAMES, args.hybrid_report
    )
    hybrid_cm = find_file(
        HYBRID_RUN, CM_NAMES, args.hybrid_cm
    )

    required_files = {
        "Baseline validation report": baseline_report,
        "Baseline validation confusion matrix": baseline_cm,
        "Hybrid validation report": hybrid_report,
        "Hybrid validation confusion matrix": hybrid_cm,
    }

    missing = [
        name for name, path in required_files.items()
        if path is None
    ]

    if missing:
        print("\nERROR: Required saved validation artifacts were not found.")
        for name in missing:
            print(f"  - {name}")

        print("\nPlease locate the existing validation CSV files and rerun")
        print("with --baseline-report, --baseline-cm, --hybrid-report,")
        print("and --hybrid-cm. Do not supply test-set files.")
        raise SystemExit(2)

    print("\nUsing these existing validation artifacts:")
    for name, path in required_files.items():
        print(f"{name}: {path}")

    # Read-only loading: no training, fitting, or artifact modification.
    baseline_report_df = load_classification_report(baseline_report)
    hybrid_report_df = load_classification_report(hybrid_report)

    baseline_matrix = load_confusion_matrix(baseline_cm)
    hybrid_matrix = load_confusion_matrix(hybrid_cm)

    comparison = compare_reports(
        baseline_report_df,
        hybrid_report_df,
    )

    pair_results = (
        pair_error_analysis(baseline_matrix, "Baseline CNN")
        + pair_error_analysis(hybrid_matrix, "Hybrid CNN + SVM")
    )
    pair_df = pd.DataFrame(pair_results)

    # Create a unique output folder. Never overwrite a previous analysis.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / f"validation_per_class_analysis_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    comparison_path = output_dir / "baseline_vs_hybrid_per_class.csv"
    pair_path = output_dir / "targeted_pair_confusions.csv"
    summary_path = output_dir / "analysis_summary.json"
    text_path = output_dir / "analysis_summary.txt"

    comparison.to_csv(comparison_path, index=False)
    pair_df.to_csv(pair_path, index=False)

    # Largest improvements and regressions are based ONLY on validation.
    best_improvements = comparison.nlargest(5, "f1_change")
    largest_regressions = comparison.nsmallest(5, "f1_change")

    summary = {
        "analysis_split": "validation",
        "baseline_report": str(baseline_report),
        "baseline_confusion_matrix": str(baseline_cm),
        "hybrid_report": str(hybrid_report),
        "hybrid_confusion_matrix": str(hybrid_cm),
        "number_of_classes_compared": int(len(comparison)),
        "mean_baseline_class_f1": float(
            comparison["f1-score_baseline"].mean()
        ),
        "mean_hybrid_class_f1": float(
            comparison["f1-score_hybrid"].mean()
        ),
        "mean_class_f1_change": float(
            comparison["f1_change"].mean()
        ),
        "best_f1_improvements": best_improvements[
            ["class_name", "f1-score_baseline",
             "f1-score_hybrid", "f1_change"]
        ].to_dict(orient="records"),
        "largest_f1_regressions": largest_regressions[
            ["class_name", "f1-score_baseline",
             "f1-score_hybrid", "f1_change"]
        ].to_dict(orient="records"),
        "targeted_pair_confusions": pair_results,
        "test_set_used": False,
        "baseline_retrained": False,
        "svm_retrained": False,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    lines = [
        "VALIDATION-ONLY PER-CLASS ANALYSIS",
        "=" * 60,
        f"Classes compared: {len(comparison)}",
        "",
        "Mean per-class F1:",
        f"Baseline CNN: {summary['mean_baseline_class_f1']:.4f}",
        f"Hybrid CNN + SVM: {summary['mean_hybrid_class_f1']:.4f}",
        f"Mean F1 change: {summary['mean_class_f1_change']:+.4f}",
        "",
        "TOP FIVE F1 IMPROVEMENTS",
        best_improvements[
            ["class_name", "f1-score_baseline",
             "f1-score_hybrid", "f1_change"]
        ].to_string(index=False),
        "",
        "TOP FIVE F1 REGRESSIONS",
        largest_regressions[
            ["class_name", "f1-score_baseline",
             "f1-score_hybrid", "f1_change"]
        ].to_string(index=False),
        "",
        "TARGETED VALIDATION CONFUSIONS",
        pair_df.to_string(index=False),
        "",
        "No test-set files were loaded.",
        "No models were trained or fitted.",
        "Existing artifacts were not overwritten.",
    ]

    text_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines))
    print("\nAnalysis saved to:")
    print(output_dir)
    print(f"\nPer-class comparison: {comparison_path.name}")
    print(f"Targeted confusions: {pair_path.name}")
    print(f"JSON summary: {summary_path.name}")
    print(f"Text summary: {text_path.name}")


if __name__ == "__main__":
    main()
