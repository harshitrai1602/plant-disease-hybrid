# Plant Disease Hybrid

A hybrid plant disease classification project using **PlantVillage** and **PlantDoc** datasets.

The project is designed to combine controlled plant-leaf imagery from PlantVillage with more diverse real-world imagery from PlantDoc and develop a robust multi-class plant disease classification system.

## Current Project Status

### Dataset Preparation — COMPLETED

The current dataset preparation pipeline has been completed.

| Component | Status |
|---|---|
| PlantVillage collection | ✅ Completed |
| PlantDoc collection | ✅ Completed |
| PlantDoc extraction | ✅ Completed |
| Dataset class mapping | ✅ Completed |
| PlantVillage canonical manifest | ✅ Completed |
| PlantDoc mapping | ✅ Completed |
| Hybrid dataset construction | ✅ Completed |
| Missing-file validation | ✅ Completed |
| Exact duplicate detection | ✅ Completed |
| Stratified dataset splitting | ✅ Completed |
| Model training | ⏳ Not started |
| Deep feature extraction | ⏳ Not started |
| Model comparison | ⏳ Not started |
| Evaluation | ⏳ Not started |
| Backend API | ⏳ Not started/under development |
| Frontend | ⏳ Planned for later |
| Deployment | ⏳ Planned for later |

## Final Hybrid Dataset

The final unified dataset contains:

- **Total images:** 164,501
- **Total classes:** 38
- **PlantVillage images:** 162,853
- **PlantDoc images:** 1,648
- **Missing files:** 0
- **Exact duplicate images removed:** 63

### Dataset Split

A stratified 70/15/15 split has been created.

| Split | Images | Classes |
|---|---:|---:|
| Training | 115,150 | 38 |
| Validation | 24,675 | 38 |
| Test | 24,676 | 38 |
| **Total** | **164,501** | **38** |

All 38 classes are present in the training, validation, and test sets.

## Dataset Sources

### PlantVillage

PlantVillage provides the majority of the images in the hybrid dataset and contains images covering multiple crops and disease categories.

### PlantDoc

PlantDoc contributes real-world plant disease images with more varied backgrounds and imaging conditions.

Only PlantDoc classes that could be reliably mapped to the project's canonical taxonomy were included.

## Canonical Classes

The current hybrid dataset contains 38 classes:

```text
Orange_HLB
Tomato_Yellow_Leaf_Curl_Virus
Soybean_healthy
Peach_Bacterial_spot
Tomato_Bacterial_spot
Tomato_Late_blight
Squash_Powdery_mildew
Tomato_Septoria_leaf_spot
Tomato_Spider_mites
Apple_healthy
Tomato_healthy
Blueberry_healthy
Pepper_healthy
Tomato_Target_Spot
Grape_Esca
Corn_Common_rust
Grape_Black_rot
Corn_healthy
Strawberry_Leaf_scorch
Grape_Leaf_blight
Cherry_Powdery_mildew
Corn_Northern_Leaf_Blight
Potato_Early_blight
Potato_Late_blight
Tomato_Early_blight
Pepper_Bacterial_spot
Tomato_Leaf_Mold
Cherry_healthy
Apple_Scab
Apple_Black_rot
Corn_Gray_leaf_spot
Strawberry_healthy
Grape_healthy
Tomato_Tomato_mosaic_virus
Raspberry_healthy
Peach_healthy
Apple_Cedar_apple_rust
Potato_healthy
```

## Important Dataset Files

### Unified Dataset

```text
data/processed/metadata/unified_dataset.csv
```

Contains the unified metadata for the final 164,501-image dataset.

Important columns:

```text
image_path
source_dataset
raw_class
label
class_id
```

### Class Mapping

```text
data/processed/metadata/class_mapping.csv
```

Contains the canonical class mapping.

### Dataset Statistics

```text
data/processed/metadata/hybrid_dataset_statistics.csv
```

Contains dataset/class statistics.

### Duplicate Report

```text
data/processed/metadata/unified_duplicates.csv
```

Contains information about duplicate-image detection.

### Final Splits

```text
data/processed/splits/train.csv
data/processed/splits/val.csv
data/processed/splits/test.csv
```

These are the final stratified 70/15/15 splits.

## Project Structure

```text
plant-disease-hybrid/
│
├── backend/
│   └── ml_api/
│       ├── inference.py
│       └── main.py
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   │   └── # Datasets - NOT committed to Git
│   │
│   └── processed/
│       ├── metadata/
│       │   ├── unified_dataset.csv
│       │   ├── class_mapping.csv
│       │   ├── hybrid_dataset_statistics.csv
│       │   └── unified_duplicates.csv
│       │
│       └── splits/
│           ├── train.csv
│           ├── val.csv
│           └── test.csv
│
├── evaluation/
│   └── reports/
│
├── models/
│   └── cnn/
│
├── scripts/
│   └── inspect_dataset.py
│
├── src/
│   ├── data/
│   │   ├── clean_dataset.py
│   │   └── prepare_splits.py
│   │
│   └── models/
│       ├── cnn_backbones.py
│       ├── extract_deep_features.py
│       └── train_cnn.py
│
├── .gitignore
└── README.md
```

## Environment

The project is currently being developed on Windows using PowerShell.

Recommended environment:

```text
Python 3.14
PyTorch with CUDA support
NVIDIA GPU
```

The current development machine has an NVIDIA GeForce RTX 3050 Laptop GPU with CUDA support.

A Python virtual environment is used:

```text
.venv/
```

The virtual environment must not be committed to Git.

## Current ML Pipeline

The intended pipeline is:

```text
PlantVillage + PlantDoc
          ↓
Dataset Cleaning
          ↓
Class Mapping
          ↓
Duplicate Detection
          ↓
Unified Hybrid Dataset
          ↓
Stratified 70/15/15 Split
          ↓
CNN Training
          ↓
Deep Feature Extraction
          ↓
Feature-Based / Hybrid Classification
          ↓
Model Evaluation
          ↓
Inference API
          ↓
Frontend
          ↓
Deployment
```

## Next Task

The immediate next task is **model training**.

Do not rebuild the dataset unless a problem is discovered in validation.

The final splits are already available:

```text
data/processed/splits/train.csv
data/processed/splits/val.csv
data/processed/splits/test.csv
```

The next developer should first verify that these files contain:

```text
Train      : 115150
Validation : 24675
Test       : 24676
Classes    : 38
```

and that all image paths exist.

After verification:

1. Configure the CNN backbone.
2. Start training using the training split.
3. Monitor validation loss and accuracy.
4. Save the best checkpoint.
5. Evaluate the trained model on the test split.
6. Generate classification metrics.
7. Generate confusion matrix and per-class performance.
8. Perform deep feature extraction if required by the project design.
9. Compare the planned classification approaches.
10. Document the final results.

## Important Git Rules

Do **not** commit:

```text
data/raw/
.venv/
__pycache__/
*.pyc
*.pth
*.pt
*.ckpt
```

The raw datasets should remain on the local machine or be obtained separately by each team member.

The Git repository should contain the code, configuration, metadata, split definitions, reports, and documentation required to reproduce the pipeline.

## Team Handoff

The dataset preparation stage is complete.

The next team member can clone this repository and continue from the **model-training stage** rather than downloading/rebuilding the dataset pipeline from the beginning.

Before training, verify the local dataset paths because the CSV files contain paths to the local dataset files.

## Project Development Status

### Minor Project / Seminar Stage

The following work has been completed:

- Hybrid dataset collection
- PlantVillage integration
- PlantDoc integration
- Dataset taxonomy/class mapping
- Duplicate detection
- Dataset validation
- Final 38-class dataset
- Stratified train/validation/test split

### Major Project — Remaining

The following stages are planned for the major project:

- CNN model training
- Model optimization
- Deep feature extraction
- Hybrid classification approach
- Model comparison
- Detailed evaluation
- Explainability/visualization if required
- ML inference pipeline
- Backend API
- Frontend application
- Integration
- Deployment

## Status

**Current milestone: Dataset Preparation COMPLETE**

**Next milestone: CNN Model Training**
