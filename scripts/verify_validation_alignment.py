
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parent.parent

VAL_CSV = ROOT / "data/processed/splits/val.csv"

BASELINE_REPORT = (
    ROOT / "outputs/baseline_cnn_retrained/val_classification_report.csv"
)
HYBRID_REPORT = (
    ROOT / "outputs/hybrid_svm_run_20261029_125118/val_classification_report.csv"
)

BASELINE_CM = (
    ROOT / "outputs/baseline_cnn_retrained/val_confusion_matrix.csv"
)
HYBRID_CM = (
    ROOT / "outputs/hybrid_svm_run_20261029_125118/val_confusion_matrix.csv"
)


def load_report(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df


def load_confusion_matrix(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def main():
    print("=" * 65)
    print("VALIDATION ALIGNMENT VERIFICATION")
    print("=" * 65)

    required = [
        VAL_CSV, BASELINE_REPORT, HYBRID_REPORT,
        BASELINE_CM, HYBRID_CM
    ]

    missing = [p for p in required if not p.exists()]

    if missing:
        print("ERROR: Missing files:")
        for p in missing:
            print(p)
        sys.exit(1)

    val = pd.read_csv(VAL_CSV)
    baseline = load_report(BASELINE_REPORT)
    hybrid = load_report(HYBRID_REPORT)
    baseline_cm = load_confusion_matrix(BASELINE_CM)
    hybrid_cm = load_confusion_matrix(HYBRID_CM)

    # Validation dataset checks
    print("\n1. VALIDATION DATASET")
    print("Images:", len(val))
    print("Unique class IDs:", val["class_id"].nunique())
    print("Unique class names:", val["label"].nunique())

    val_support = (
        val.groupby("label")
        .size()
        .sort_index()
    )

    # Extract class names from reports
    excluded = {"accuracy", "macro avg", "weighted avg", "micro avg", "samples avg"}

    baseline_classes = sorted(
        x for x in baseline.index if x not in excluded
    )
    hybrid_classes = sorted(
        x for x in hybrid.index if x not in excluded
    )

    expected_classes = sorted(val["label"].astype(str).unique())

    print("\n2. CLASS MAPPING")
    print("Validation classes:", len(expected_classes))
    print("Baseline report classes:", len(baseline_classes))
    print("Hybrid report classes:", len(hybrid_classes))

    print(
        "Baseline class names match validation:",
        baseline_classes == expected_classes
    )
    print(
        "Hybrid class names match validation:",
        hybrid_classes == expected_classes
    )

    # Check per-class support
    print("\n3. PER-CLASS SUPPORT")

    support_ok = True

    for name in expected_classes:
        expected = int(val_support[name])

        b_support = int(round(float(baseline.loc[name, "support"])))
        h_support = int(round(float(hybrid.loc[name, "support"])))

        if expected != b_support or expected != h_support:
            support_ok = False
            print(
                f"MISMATCH: {name}: "
                f"CSV={expected}, Baseline={b_support}, Hybrid={h_support}"
            )

    if support_ok:
        print("PASS: All 38 classes have matching support counts.")

    # Check confusion matrices
    print("\n4. CONFUSION MATRIX ALIGNMENT")

    b_rows = sorted(baseline_cm.index)
    b_cols = sorted(baseline_cm.columns)
    h_rows = sorted(hybrid_cm.index)
    h_cols = sorted(hybrid_cm.columns)

    cm_labels_ok = (
        b_rows == expected_classes
        and b_cols == expected_classes
        and h_rows == expected_classes
        and h_cols == expected_classes
    )

    print("Baseline matrix labels match:", b_rows == expected_classes and b_cols == expected_classes)
    print("Hybrid matrix labels match:", h_rows == expected_classes and h_cols == expected_classes)

    # Check matrix row totals against validation support
    row_support_ok = True

    for name in expected_classes:
        b_total = int(baseline_cm.loc[name].sum())
        h_total = int(hybrid_cm.loc[name].sum())
        expected = int(val_support[name])

        if b_total != expected or h_total != expected:
            row_support_ok = False
            print(
                f"ROW TOTAL MISMATCH: {name}: "
                f"CSV={expected}, Baseline={b_total}, Hybrid={h_total}"
            )

    if row_support_ok:
        print("PASS: Confusion matrix row totals match validation support.")

    # Overall totals
    print("\n5. TOTAL IMAGE COUNTS")
    print("Validation CSV:", len(val))
    print("Baseline report support:", int(baseline.loc[expected_classes, "support"].sum()))
    print("Hybrid report support:", int(hybrid.loc[expected_classes, "support"].sum()))
    print("Baseline confusion matrix:", int(baseline_cm.to_numpy().sum()))
    print("Hybrid confusion matrix:", int(hybrid_cm.to_numpy().sum()))

    print("\n" + "=" * 65)

    if (
        len(val) == 24675
        and len(expected_classes) == 38
        and baseline_classes == expected_classes
        and hybrid_classes == expected_classes
        and support_ok
        and cm_labels_ok
        and row_support_ok
    ):
        print("RESULT: STRUCTURAL ALIGNMENT CHECK PASSED")
        print("Both reports and matrices agree with the validation CSV.")
        print("Note: This does not independently prove identical image-level predictions inputs.")
    else:
        print("RESULT: CHECK FAILED OR INCOMPLETE")
        print("Review the mismatches above before relying on the comparison.")

    print("=" * 65)


if __name__ == "__main__":
    main()
