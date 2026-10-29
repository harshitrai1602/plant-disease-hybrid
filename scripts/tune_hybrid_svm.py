
from pathlib import Path
from datetime import datetime
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Locate the existing hybrid run
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUN_DIR = PROJECT_ROOT / "outputs" / "hybrid_svm_run_20261029_125118"

if not RUN_DIR.exists():
    raise FileNotFoundError(
        f"Existing hybrid run directory not found: {RUN_DIR}"
    )

# Store everything in a NEW directory.
# Do not overwrite the original hybrid results.
experiment_name = datetime.now().strftime(
    "svm_tuning_%Y%m%d_%H%M%S"
)
OUTPUT_DIR = RUN_DIR / experiment_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

print("Project:", PROJECT_ROOT)
print("Features:", RUN_DIR)
print("New experiment output:", OUTPUT_DIR)


# --------------------------------------------------
# 2. Load previously extracted CNN features
# --------------------------------------------------

def load_features(filename):
    path = RUN_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")

    with np.load(path) as data:
        X = data["X"].astype(np.float32, copy=False)
        y = data["y"].astype(np.int64, copy=False)

    print(
        f"{filename}: X={X.shape}, y={y.shape}, "
        f"features dtype={X.dtype}"
    )

    if len(X) != len(y):
        raise ValueError(f"Feature/label count mismatch in {filename}")

    return X, y


X_train, y_train = load_features("train_features.npz")
X_val, y_val = load_features("val_features.npz")
X_test, y_test = load_features("test_features.npz")


# --------------------------------------------------
# 3. Read class names from the existing run summary
# --------------------------------------------------

summary_path = RUN_DIR / "run_summary.json"

with open(summary_path, "r", encoding="utf-8") as file:
    summary = json.load(file)

class_names_by_id = summary["class_names_by_id"]

class_ids = sorted(int(class_id) for class_id in class_names_by_id)
class_names = [
    class_names_by_id[str(class_id)]
    for class_id in class_ids
]

if set(np.unique(y_train)) != set(class_ids):
    raise ValueError(
        "Training labels do not match the class IDs in run_summary.json"
    )

if set(np.unique(y_val)) - set(class_ids):
    raise ValueError("Validation data contains unknown class IDs.")

if set(np.unique(y_test)) - set(class_ids):
    raise ValueError("Test data contains unknown class IDs.")


# --------------------------------------------------
# 4. Compare C values using validation data only
# --------------------------------------------------

C_VALUES = [0.1, 1.0, 10.0]
comparison_rows = []
validation_models = {}

for C in C_VALUES:
    print("\n" + "=" * 60)
    print(f"Training LinearSVC with C={C}")
    print("=" * 60)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(
            C=C,
            class_weight="balanced",
            random_state=42,
            max_iter=5000,
        )),
    ])

    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, val_predictions)

    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_val,
            val_predictions,
            labels=class_ids,
            average="macro",
            zero_division=0,
        )
    )

    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_val,
        val_predictions,
        labels=class_ids,
        average="weighted",
        zero_division=0,
    )

    print(f"C:                 {C}")
    print(f"Validation accuracy: {accuracy:.4%}")
    print(f"Macro precision:     {precision:.4%}")
    print(f"Macro recall:        {recall:.4%}")
    print(f"Macro F1:            {macro_f1:.4%}")
    print(f"Weighted F1:         {weighted_f1:.4%}")

    comparison_rows.append({
        "C": C,
        "validation_accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    })

    # Save validation predictions/report separately for each C.
    report = classification_report(
        y_val,
        val_predictions,
        labels=class_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / f"validation_report_C_{C}.csv"
    )

    validation_models[C] = model

    del val_predictions


# --------------------------------------------------
# 5. Select the best C using validation macro F1
# --------------------------------------------------

comparison_df = pd.DataFrame(comparison_rows)

comparison_df = comparison_df.sort_values(
    by=["macro_f1", "validation_accuracy"],
    ascending=False,
)

comparison_df.to_csv(
    OUTPUT_DIR / "validation_C_comparison.csv",
    index=False,
)

print("\n" + "=" * 60)
print("VALIDATION COMPARISON")
print("=" * 60)
print(comparison_df.to_string(index=False))

best_C = float(comparison_df.iloc[0]["C"])
best_model = validation_models[best_C]

print(f"\nSelected C={best_C}")
print("Selection criterion: highest validation macro F1")
print("Test data has not been used for model selection.")


# --------------------------------------------------
# 6. Save the selected model in the new experiment folder
# --------------------------------------------------

model_path = OUTPUT_DIR / "best_hybrid_svm.joblib"
joblib.dump(best_model, model_path)

with open(
    OUTPUT_DIR / "selected_configuration.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "selected_C": best_C,
            "selection_metric": "validation_macro_f1",
            "C_values_compared": C_VALUES,
            "class_names": class_names,
        },
        file,
        indent=2,
    )


# --------------------------------------------------
# 7. Evaluate the selected model on the test set ONCE
# --------------------------------------------------

print("\nEvaluating selected model on test data...")

test_predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, test_predictions)

test_precision, test_recall, test_macro_f1, _ = (
    precision_recall_fscore_support(
        y_test,
        test_predictions,
        labels=class_ids,
        average="macro",
        zero_division=0,
    )
)

_, _, test_weighted_f1, _ = precision_recall_fscore_support(
    y_test,
    test_predictions,
    labels=class_ids,
    average="weighted",
    zero_division=0,
)

test_metrics = {
    "selected_C": best_C,
    "test_accuracy": float(test_accuracy),
    "test_macro_precision": float(test_precision),
    "test_macro_recall": float(test_recall),
    "test_macro_f1": float(test_macro_f1),
    "test_weighted_f1": float(test_weighted_f1),
}

with open(
    OUTPUT_DIR / "selected_model_test_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(test_metrics, file, indent=2)

report = classification_report(
    y_test,
    test_predictions,
    labels=class_ids,
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(report).transpose().to_csv(
    OUTPUT_DIR / "selected_model_test_report.csv"
)

matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=class_ids,
)

pd.DataFrame(
    matrix,
    index=class_names,
    columns=class_names,
).to_csv(OUTPUT_DIR / "selected_model_test_confusion_matrix.csv")

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)
print(f"Selected C:       {best_C}")
print(f"Test accuracy:    {test_accuracy:.4%}")
print(f"Macro precision:  {test_precision:.4%}")
print(f"Macro recall:     {test_recall:.4%}")
print(f"Macro F1:         {test_macro_f1:.4%}")
print(f"Weighted F1:      {test_weighted_f1:.4%}")

print("\nSaved all new experiment outputs to:")
print(OUTPUT_DIR)
