
from pathlib import Path
import pandas as pd
import sys

# Use the run directory from your successful experiment.
run_dir = Path(
    sys.argv[1] if len(sys.argv) > 1
    else "outputs/hybrid_svm_run_20261029_125118"
)

report_path = run_dir / "test_classification_report.csv"
matrix_path = run_dir / "test_confusion_matrix.csv"

if not report_path.exists() or not matrix_path.exists():
    raise FileNotFoundError(
        f"Could not find the report or confusion matrix in {run_dir}"
    )

report = pd.read_csv(report_path, index_col=0)
cm = pd.read_csv(matrix_path, index_col=0)

# Keep individual disease classes, excluding aggregate rows.
classes = report.loc[
    ~report.index.isin(["accuracy", "macro avg", "weighted avg"])
].copy()

for col in ["precision", "recall", "f1-score", "support"]:
    classes[col] = pd.to_numeric(classes[col], errors="coerce")

classes = classes.dropna(subset=["f1-score"])

print("\n=== 15 CLASSES WITH LOWEST F1 ===")
print(
    classes.sort_values("f1-score").head(15)[
        ["precision", "recall", "f1-score", "support"]
    ].to_string(float_format=lambda x: f"{x:.4f}")
)

print("\n=== 15 CLASSES WITH LOWEST RECALL ===")
print(
    classes.sort_values("recall").head(15)[
        ["precision", "recall", "f1-score", "support"]
    ].to_string(float_format=lambda x: f"{x:.4f}")
)

# Find the largest off-diagonal confusion-matrix entries.
pairs = []
for i, actual in enumerate(cm.index):
    for j, predicted in enumerate(cm.columns):
        count = int(cm.iloc[i, j])
        if i != j and count > 0:
            pairs.append((count, str(actual), str(predicted)))

pairs.sort(reverse=True)

print("\n=== 15 MOST COMMON MISCLASSIFICATIONS ===")
print("Count | Actual class -> Predicted class")
for count, actual, predicted in pairs[:15]:
    print(f"{count:5d} | {actual} -> {predicted}")

# Save a separate analysis report without altering original files.
out = run_dir / "per_class_analysis.csv"
classes.sort_values("f1-score").to_csv(out)
print(f"\nPer-class analysis saved to: {out}")
