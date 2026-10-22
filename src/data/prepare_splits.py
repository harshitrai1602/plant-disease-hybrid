from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = Path("data/processed/metadata/unified_dataset.csv")
OUT_DIR = Path("data/processed/splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CREATING FINAL 38-CLASS HYBRID DATASET SPLITS")
print("=" * 60)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Unified dataset not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required = ["image_path", "source_dataset", "raw_class", "label", "class_id"]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

print(f"Total images: {len(df)}")
print(f"Total classes: {df['label'].nunique()}")

# ---------------------------------------------------------
# Remove missing files, just in case
# ---------------------------------------------------------
print("\nChecking file existence...")

exists = df["image_path"].map(lambda x: Path(x).exists())
missing_count = (~exists).sum()

print(f"Existing files: {exists.sum()}")
print(f"Missing files: {missing_count}")

if missing_count > 0:
    print("Removing missing files...")
    df = df[exists].copy()

# ---------------------------------------------------------
# Remove exact duplicate image paths
# ---------------------------------------------------------
before = len(df)
df = df.drop_duplicates(subset=["image_path"]).copy()

print(f"\nDuplicate paths removed: {before - len(df)}")

# ---------------------------------------------------------
# Verify class counts
# ---------------------------------------------------------
counts = df["label"].value_counts()

print("\nClass distribution:")
print(counts.to_string())

rare = counts[counts < 3]

if len(rare) > 0:
    print("\nWARNING: Classes with fewer than 3 images:")
    print(rare.to_string())
    print("\nThese classes cannot safely be stratified.")
    df = df[~df["label"].isin(rare.index)].copy()

# ---------------------------------------------------------
# Recreate class IDs consistently
# ---------------------------------------------------------
classes = sorted(df["label"].unique())

class_to_id = {label: i for i, label in enumerate(classes)}

df["class_id"] = df["label"].map(class_to_id)

print("\nFINAL DATASET")
print(f"Images : {len(df)}")
print(f"Classes: {len(classes)}")

# ---------------------------------------------------------
# 70% TRAIN / 15% VALIDATION / 15% TEST
# ---------------------------------------------------------
print("\nCreating stratified 70/15/15 split...")

train, temp = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

val, test = train_test_split(
    temp,
    test_size=0.50,
    stratify=temp["label"],
    random_state=42
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
train_file = OUT_DIR / "train.csv"
val_file = OUT_DIR / "val.csv"
test_file = OUT_DIR / "test.csv"

train.to_csv(train_file, index=False)
val.to_csv(val_file, index=False)
test.to_csv(test_file, index=False)

print("\n" + "=" * 60)
print("FINAL SPLIT COMPLETE")
print("=" * 60)

print(f"Train      : {len(train)}")
print(f"Validation : {len(val)}")
print(f"Test       : {len(test)}")
print(f"Total      : {len(train) + len(val) + len(test)}")

print("\nClasses:")
print(f"Train: {train['label'].nunique()}")
print(f"Val  : {val['label'].nunique()}")
print(f"Test : {test['label'].nunique()}")

print("\nFiles:")
print(train_file)
print(val_file)
print(test_file)

# ---------------------------------------------------------
# Verify every class exists in every split
# ---------------------------------------------------------
train_classes = set(train["label"])
val_classes = set(val["label"])
test_classes = set(test["label"])

all_classes = set(classes)

print("\nClass coverage:")

print("Train missing classes:", all_classes - train_classes)
print("Val missing classes  :", all_classes - val_classes)
print("Test missing classes :", all_classes - test_classes)

if (
    train_classes == all_classes
    and val_classes == all_classes
    and test_classes == all_classes
):
    print("\nSUCCESS: All 38 classes are present in all splits.")
else:
    print("\nWARNING: Some classes are missing from one or more splits.")

print("\nDO NOT TRAIN YET.")
