from pathlib import Path
import pandas as pandas
import hashlib

META = Path("data/processed/metadata")

PV_FILE = META / "plantvillage_canonical.csv"
PD_FILE = META / "plantdoc_manifest.csv"

OUT_FILE = META / "unified_dataset.csv"
DUP_FILE = META / "unified_duplicates.csv"

print("=" * 60)
print("BUILDING FINAL HYBRID DATASET")
print("=" * 60)

# =========================================================
# LOAD DATA
# =========================================================

pv_df = pandas.read_csv(PV_FILE)
plantdoc_df = pandas.read_csv(PD_FILE)

print("PlantVillage rows:", len(pv_df))
print("PlantDoc rows:", len(plantdoc_df))

# =========================================================
# PLANTDOC -> CANONICAL LABEL MAPPING
# =========================================================

PD_MAPPING = {

    "Apple Scab Leaf":
        "Apple_Scab",

    "Apple rust leaf":
        "Apple_Cedar_apple_rust",

    "Bell_pepper leaf spot":
        "Pepper_Bacterial_spot",

    "Corn Gray leaf spot":
        "Corn_Gray_leaf_spot",

    "Corn leaf blight":
        "Corn_Northern_Leaf_Blight",

    "Corn rust leaf":
        "Corn_Common_rust",

    "Potato leaf early blight":
        "Potato_Early_blight",

    "Potato leaf late blight":
        "Potato_Late_blight",

    "Squash Powdery mildew leaf":
        "Squash_Powdery_mildew",

    "Tomato Early blight leaf":
        "Tomato_Early_blight",

    "Tomato leaf bacterial spot":
        "Tomato_Bacterial_spot",

    "Tomato leaf late blight":
        "Tomato_Late_blight",

    "Tomato leaf mosaic virus":
        "Tomato_Tomato_mosaic_virus",

    "Tomato leaf yellow virus":
        "Tomato_Yellow_Leaf_Curl_Virus",

    "Tomato mold leaf":
        "Tomato_Leaf_Mold",

    "Tomato Septoria leaf spot":
        "Tomato_Septoria_leaf_spot",

    "Tomato two spotted spider mites leaf":
        "Tomato_Spider_mites",
}

# =========================================================
# MAP PLANTDOC
# =========================================================

plantdoc_df["label"] = plantdoc_df["raw_class"].map(PD_MAPPING)

mapped_plantdoc = plantdoc_df[
    plantdoc_df["label"].notna()
].copy()

unmapped_plantdoc = plantdoc_df[
    plantdoc_df["label"].isna()
].copy()

print()
print("PlantDoc mapped:", len(mapped_plantdoc))
print("PlantDoc excluded:", len(unmapped_plantdoc))

print()
print("Excluded PlantDoc classes:")

for cls in sorted(unmapped_plantdoc["raw_class"].unique()):
    print("  -", cls)

# =========================================================
# STANDARDIZE PLANTVILLAGE
# =========================================================

pv_df["source_dataset"] = "plantvillage"

pv_df = pv_df[
    [
        "image_path",
        "source_dataset",
        "raw_class",
        "label"
    ]
].copy()

mapped_plantdoc = mapped_plantdoc[
    [
        "image_path",
        "source_dataset",
        "raw_class",
        "label"
    ]
].copy()

# =========================================================
# COMBINE
# =========================================================

combined_df = pandas.concat(
    [pv_df, mapped_plantdoc],
    ignore_index=True
)

print()
print("Combined before duplicate removal:", len(combined_df))

# =========================================================
# VERIFY FILE EXISTENCE
# =========================================================

print()
print("Checking file existence...")

exists_mask = combined_df["image_path"].map(
    lambda x: Path(x).exists()
)

missing_df = combined_df[
    ~exists_mask
].copy()

combined_df = combined_df[
    exists_mask
].copy()

print("Missing files:", len(missing_df))
print("Existing files:", len(combined_df))

if len(missing_df) > 0:

    missing_df.to_csv(
        META / "missing_files.csv",
        index=False,
        encoding="utf-8"
    )

# =========================================================
# EXACT DUPLICATE DETECTION
# =========================================================

print()
print("Checking exact duplicate images...")

hash_to_path = {}
keep_indices = []
duplicate_rows = []

for index, path in enumerate(
    combined_df["image_path"]
):

    try:

        with open(path, "rb") as f:
            file_hash = hashlib.sha256(
                f.read()
            ).hexdigest()

    except Exception as error:

        print(
            "Could not hash:",
            path,
            "|",
            error
        )

        continue

    if file_hash in hash_to_path:

        duplicate_rows.append({
            "image_path": path,
            "duplicate_of": hash_to_path[file_hash],
            "sha256": file_hash
        })

    else:

        hash_to_path[file_hash] = path
        keep_indices.append(index)

    if (index + 1) % 5000 == 0:

        print(
            f"Hashed {index + 1}/{len(combined_df)}"
        )

combined_df = combined_df.iloc[
    keep_indices
].reset_index(drop=True)

duplicates_df = pandas.DataFrame(
    duplicate_rows
)

duplicates_df.to_csv(
    DUP_FILE,
    index=False,
    encoding="utf-8"
)

print(
    "Exact duplicates removed:",
    len(duplicate_rows)
)

# =========================================================
# CLASS IDS
# =========================================================

classes = sorted(
    combined_df["label"].dropna().unique()
)

class_to_id = {
    label: index
    for index, label in enumerate(classes)
}

combined_df["class_id"] = combined_df[
    "label"
].map(class_to_id)

# =========================================================
# CLASS MAPPING FILE
# =========================================================

class_map_df = pandas.DataFrame([
    {
        "class_id": class_to_id[label],
        "label": label
    }
    for label in classes
])

class_map_df.to_csv(
    META / "class_mapping.csv",
    index=False,
    encoding="utf-8"
)

# =========================================================
# FINAL DATASET
# =========================================================

combined_df.to_csv(
    OUT_FILE,
    index=False,
    encoding="utf-8"
)

# =========================================================
# DATASET STATISTICS
# =========================================================

statistics_df = (
    combined_df
    .groupby(
        ["label", "source_dataset"]
    )
    .size()
    .reset_index(name="count")
    .sort_values(
        ["label", "source_dataset"]
    )
)

statistics_df.to_csv(
    META / "hybrid_dataset_statistics.csv",
    index=False,
    encoding="utf-8"
)

# =========================================================
# FINAL REPORT
# =========================================================

print()
print("=" * 60)
print("FINAL HYBRID DATASET")
print("=" * 60)

print(
    "Total images:",
    len(combined_df)
)

print(
    "Total classes:",
    len(classes)
)

print()
print("Images by source:")

print(
    combined_df[
        "source_dataset"
    ].value_counts().to_string()
)

print()
print("Images by class:")

print(
    combined_df[
        "label"
    ]
    .value_counts()
    .sort_values(
        ascending=False
    )
    .to_string()
)

print()
print("=" * 60)
print("FILES CREATED")
print("=" * 60)

print(
    "Unified dataset:",
    OUT_FILE
)

print(
    "Class mapping:",
    META / "class_mapping.csv"
)

print(
    "Statistics:",
    META / "hybrid_dataset_statistics.csv"
)

print(
    "Duplicates:",
    DUP_FILE
)

if len(missing_df) > 0:

    print(
        "Missing files:",
        META / "missing_files.csv"
    )

print()
print("=" * 60)
print("HYBRID DATASET BUILD COMPLETE")
print("=" * 60)

print("DO NOT TRAIN YET.")
