# Plant Disease Hybrid Classification

A multi-class plant disease recognition project combining **PlantVillage** and **PlantDoc** datasets. The project uses a PyTorch Convolutional Neural Network (CNN) as a baseline and a hybrid **CNN feature extractor + Support Vector Machine (SVM)** classifier for comparison.

The unified dataset contains 164,501 images across 38 canonical plant health and disease classes.

## Current Project Status

| Component                                  | Status                                                             |
| ------------------------------------------ | ------------------------------------------------------------------ |
| PlantVillage and PlantDoc collection       | Completed                                                          |
| Dataset class mapping and unification      | Completed                                                          |
| Missing-file validation                    | Completed                                                          |
| Exact duplicate detection                  | Completed                                                          |
| Stratified train/validation/test splitting | Completed                                                          |
| Dataset and DataLoader verification        | Completed                                                          |
| Image preprocessing and augmentation tests | Completed                                                          |
| Baseline CNN training                      | 20 epochs recorded                                                 |
| Baseline checkpoint                        | Saved                                                              |
| Hybrid CNN feature extraction              | Completed                                                          |
| Initial CNN + SVM model                    | Completed                                                          |
| SVM hyperparameter comparison              | Completed                                                          |
| Hybrid test evaluation                     | Completed                                                          |
| Baseline held-out test evaluation          | Verify and document separately                                     |
| Baseline-versus-hybrid comparison          | Preliminary comparison available; verify preprocessing consistency |
| Inference API                              | Pending                                                            |
| Frontend integration                       |  Not Planned yet                                                            |
| Deployment                                 |  Not Planned yet                                                            |

## 1. Project Overview

The objective is to classify plant leaf images into 38 canonical plant health and disease categories.

PlantVillage contributes predominantly controlled-condition images, while PlantDoc introduces greater variation in backgrounds and image-capture conditions.

The project workflow includes:

1. Collecting and validating the datasets.
2. Mapping source labels to a unified taxonomy.
3. Removing exact duplicate images.
4. Creating stratified train, validation, and test splits.
5. Training and evaluating a baseline CNN.
6. Extracting deep features using the trained CNN.
7. Training an SVM on the extracted features.
8. Selecting SVM hyperparameters using validation performance.
9. Evaluating the selected hybrid model on the reserved test split.
10. Comparing model performance and preparing application integration.

## 2. Unified Dataset

The prepared dataset contains **164,501 images across 38 classes**.

| Dataset component              | Number of images |
| ------------------------------ | ---------------: |
| PlantVillage                   |          162,853 |
| PlantDoc                       |            1,648 |
| Total images                   |          164,501 |
| Canonical classes              |               38 |
| Missing image files detected   |                0 |
| Exact duplicate images removed |               63 |

Only source classes that could be reliably mapped to the project's canonical taxonomy were included.

### Dataset split

A stratified 70/15/15 split was prepared.

| Split      |      Images | Classes |
| ---------- | ----------: | ------: |
| Training   |     115,150 |      38 |
| Validation |      24,675 |      38 |
| Test       |      24,676 |      38 |
| **Total**  | **164,501** |  **38** |

The test split is reserved for final evaluation. It must not be used for model training or hyperparameter selection.

## 3. Canonical Classes

The dataset contains these 38 classes:

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

## 4. Important Dataset Files

### Unified metadata

```text
data/processed/metadata/unified_dataset.csv
```

Contains image paths, source datasets, original class labels, canonical labels, and class IDs.

Important columns include:

- `image_path`
- `source_dataset`
- `raw_class`
- `label`
- `class_id`

### Other metadata

```text
data/processed/metadata/class_mapping.csv
data/processed/metadata/hybrid_dataset_statistics.csv
data/processed/metadata/unified_duplicates.csv
```

These files contain label mappings, dataset statistics, and duplicate-detection information.

### Data splits

```text
data/processed/splits/train.csv
data/processed/splits/val.csv
data/processed/splits/test.csv
```

These files define the training, validation, and test sets. Keep them unchanged when comparing the baseline and hybrid models.

## 5. Development Environment

The project uses Windows, Visual Studio Code, PowerShell, Python, PyTorch, and scikit-learn.

| Component              | Configuration                                  |
| ---------------------- | ---------------------------------------------- |
| Operating system       | Windows                                        |
| Python                 | 3.14.5, as recorded in project documentation   |
| PyTorch                | Version recorded by the experiment environment |
| Torchvision            | Version recorded by the experiment environment |
| Input resolution       | 224 × 224                                      |
| Number of classes      | 38                                             |
| Baseline optimizer     | Adam                                           |
| Baseline loss          | Cross-entropy                                  |
| Baseline learning rate | 0.001                                          |
| Baseline batch size    | 4                                              |
| Hybrid classifier      | LinearSVC                                      |
| SVM feature scaling    | StandardScaler                                 |

**Hardware note:** Record the actual device used for each experiment. CUDA availability and the GPU used during training should be verified from that run's logs rather than assumed to be identical across computers.

The virtual environment is local to each machine and should not be committed to GitHub.

## 6. Dataset and Preprocessing Verification

The dataset and preprocessing pipeline was tested before full training.

The verification covered:

- Loading the split CSV files.
- Checking image paths.
- Confirming class coverage across splits.
- Loading images and labels through PyTorch DataLoaders.
- Checking image tensor shape `(3, 224, 224)`.
- Checking image tensor dtype `torch.float32`.
- Checking label tensor dtype `torch.int64`.
- Checking tensor values for finite numbers.
- Executing training and evaluation transformations.
- Verifying model forward and backward passes.

The project also recorded successful dataset and preprocessing checks.

## 7. Baseline CNN Architecture

The baseline is a custom PyTorch CNN named `BaselineCNN`.

### Architecture

The network contains four convolutional stages with output channels:

1. 32 channels
2. 64 channels
3. 128 channels
4. 256 channels

The architecture uses convolutional layers, batch normalization, ReLU activations, and max pooling in the earlier stages. Adaptive average pooling produces a fixed-size feature representation.

The classification head consists of:

- Flattening
- Dropout with probability 0.3
- A fully connected layer producing 38 class scores

The model contains approximately **399,142 trainable parameters**, according to the project architecture record.

### Training configuration

| Parameter         | Value         |
| ----------------- | ------------- |
| Model             | BaselineCNN   |
| Training images   | 115,150       |
| Validation images | 24,675        |
| Recorded epochs   | 20            |
| Batch size        | 4             |
| Learning rate     | 0.001         |
| Optimizer         | Adam          |
| Loss function     | Cross-entropy |
| Input resolution  | 224 × 224     |

## 8. Full-Dataset Baseline CNN Training

The baseline CNN was trained for **20 recorded epochs**. The training history contains training loss, training accuracy, validation loss, validation accuracy, and elapsed time per epoch.

Training history:

```text
outputs/baseline_cnn_retrained/training_history.csv
```

### Complete training history

| Epoch | Train Loss | Train Accuracy |      Val Loss | Val Accuracy | Duration (s) |
| ----: | ---------: | -------------: | ------------: | -----------: | -----------: |
|     1 |  1.4561616 |         58.18% |     0.7468221 |       78.75% |      7431.21 |
|     2 |  0.7605283 |         76.91% |     0.4972142 |       85.59% |      6644.92 |
|     3 |  0.5638184 |         82.85% |     0.3969425 |       88.58% |      6407.29 |
|     4 |  0.4580436 |         85.87% |     0.3329189 |       90.04% |      6516.69 |
|     5 |  0.3939223 |         87.85% |     0.2599354 |       92.31% |      6501.87 |
|     6 |  0.3539882 |         89.06% |     0.2971917 |       90.97% |      6458.10 |
|     7 |  0.3193868 |         90.07% |     0.2168564 |       93.25% |      6444.67 |
|     8 |  0.2922549 |         90.99% |     0.1886616 |       94.43% |      6524.96 |
|     9 |  0.2714475 |         91.52% |     0.2205243 |       93.17% |      7240.53 |
|    10 |  0.2572454 |         92.04% |     0.2009383 |       93.81% |      6666.68 |
|    11 |  0.2410960 |         92.42% |     0.1611719 |       95.08% |      7620.03 |
|    12 |  0.2315910 |         92.77% |     0.1536519 |       95.43% |      6574.57 |
|    13 |  0.2203640 |         93.11% |     0.1415694 |       95.79% |      6403.88 |
|    14 |  0.2092829 |         93.43% |     0.1764299 |       94.38% |      6398.92 |
|    15 |  0.2035673 |         93.51% |     0.1334466 |       95.99% |      6389.38 |
|    16 |  0.1944735 |         93.83% |     0.1340617 |       95.91% |      6420.04 |
|    17 |  0.1875102 |         94.08% | **0.1273499** |   **96.36%** |      6568.27 |
|    18 |  0.1863113 |         94.16% |     0.1415347 |       95.68% |      6529.21 |
|    19 |  0.1790769 |         94.39% |     0.1366558 |       95.81% |      6523.25 |
|    20 |  0.1725044 |     **94.54%** |     0.1324786 |       96.02% |      6410.56 |

*Accuracy is shown as a percentage; loss and duration are rounded for readability.*

### Training results

- **Highest recorded validation accuracy:** 96.3607%, Epoch 17.
- **Lowest recorded validation loss:** 0.1273499, Epoch 17.
- **Training accuracy at Epoch 20:** 94.5428%.
- **Training loss at Epoch 20:** 0.1725044.
- **Total recorded training time:** approximately 36.9 hours.

The training loss generally decreases across the 20 epochs. Validation accuracy also improves overall, although some epochs show temporary declines. Epoch 17 has both the highest validation accuracy and the lowest validation loss in the supplied history.

The saved checkpoint should be verified to confirm that it corresponds to the intended best epoch and checkpoint-selection criterion.

### Checkpoint and outputs

The baseline checkpoint path recorded for the earlier full-training run is:

```text
outputs/baseline_cnn_full/best_model.pth
```

The later hybrid experiment used this checkpoint path:

```text
outputs/baseline_cnn_retrained/best_model.pth (this corresponds to 20 epoch training)
```

Verify which checkpoint corresponds to the 20-epoch history before treating these paths as interchangeable. Keep the original checkpoint and training history intact.

## 9. Hybrid CNN + SVM Classification

The hybrid approach uses the trained CNN as a feature extractor and a Linear Support Vector Classifier as the downstream classifier.

### Workflow

1. Load the selected CNN checkpoint.
2. Extract the 256-dimensional representation from the CNN.
3. Save feature vectors and labels for each split.
4. Standardize the features using `StandardScaler`.
5. Train a `LinearSVC` classifier on training features.
6. Compare SVM hyperparameters using validation performance.
7. Evaluate the selected configuration on the test set.

The CNN was not retrained during the initial feature-extraction and SVM tuning experiments.

### Extracted feature files

The completed hybrid run is stored at:

```text
outputs/hybrid_svm_run_20261029_125118/
```

The saved feature files are:

```text
train_features.npz
val_features.npz
test_features.npz
```

Feature dimensions:

| Split      |  Images | Feature shape   |
| ---------- | ------: | --------------- |
| Training   | 115,150 | `(115150, 256)` |
| Validation |  24,675 | `(24675, 256)`  |
| Test       |  24,676 | `(24676, 256)`  |

Each feature vector represents an image using the CNN's 256-dimensional pooled representation.

### Initial SVM configuration

| Parameter          | Value          |
| ------------------ | -------------- |
| Classifier         | LinearSVC      |
| Initial `C`        | 1.0            |
| Class weighting    | Balanced       |
| Random state       | 42             |
| Maximum iterations | 5000           |
| Feature scaling    | StandardScaler |

### Initial hybrid model results

| Metric          | Validation |     Test |
| --------------- | ---------: | -------: |
| Accuracy        |   97.4914% | 97.5361% |
| Macro precision |   96.5528% | 96.2924% |
| Macro recall    |   96.6293% | 96.7466% |
| Macro F1        |   96.5851% | 96.5064% |
| Weighted F1     |   97.4936% | 97.5401% |

The test results above are from the completed initial hybrid evaluation.

### Saved initial hybrid artifacts

The original hybrid run contains the following outputs:

```text
outputs/hybrid_svm_run_20261029_125118/
    hybrid_svm.joblib
    run_summary.json
    test_metrics.json
    test_classification_report.csv
    test_confusion_matrix.csv
    val_metrics.json
    val_classification_report.csv
    val_confusion_matrix.csv
    train_features.npz
    val_features.npz
    test_features.npz
    per_class_analysis.csv
```

The actual directory contents should be checked if any file is not present.

## 10. SVM Hyperparameter Tuning

A follow-up experiment compared three values of the SVM regularization parameter, `C`.

The experiment reused the existing CNN feature vectors. The CNN was not retrained, and the features were not extracted again.

### Validation comparison

|       C | Validation accuracy | Macro precision | Macro recall |     Macro F1 |  Weighted F1 |
| ------: | ------------------: | --------------: | -----------: | -----------: | -----------: |
|     0.1 |            97.4022% |        96.1607% |     96.6288% |     96.3868% |     97.4032% |
| **1.0** |        **97.4914%** |    **96.5528%** | **96.6293%** | **96.5851%** | **97.4936%** |
|    10.0 |            97.2847% |        96.2671% |     96.1687% |     96.2031% |     97.2871% |

The selection criterion was the highest validation macro F1, with validation accuracy used as a tie-breaker.

**Selected setting: `C = 1.0`.**

The selected setting also achieved the highest validation accuracy among the three tested values. Therefore, this experiment did not demonstrate a benefit from changing the initial SVM configuration.

### Final selected-model test metrics

The selected `C = 1.0` model was evaluated on the test split.

| Metric          | Test result |
| --------------- | ----------: |
| Accuracy        |    97.5361% |
| Macro precision |    96.2924% |
| Macro recall    |    96.7466% |
| Macro F1        |    96.5064% |
| Weighted F1     |    97.5401% |

These metrics match the initial hybrid model's reported test results.

### Tuning experiment outputs

```text
outputs/hybrid_svm_run_20261029_125118/
    svm_tuning_20261029_142949/
        validation_C_comparison.csv
        validation_report_C_0.1.csv
        validation_report_C_1.0.csv
        validation_report_C_10.0.csv
        selected_configuration.json
        best_hybrid_svm.joblib
        selected_model_test_metrics.json
        selected_model_test_report.csv
        selected_model_test_confusion_matrix.csv
```

The tuning experiment used the same existing feature files and saved its outputs in a new directory without replacing the original artifacts.

## 11. Preliminary Baseline-versus-Hybrid Comparison

The current recorded metrics are:

| Metric      | Baseline CNN | Hybrid CNN + SVM |
| ----------- | -----------: | ---------------: |
| Accuracy    |     96.0204% |         97.5361% |
| Macro F1    |     94.3328% |         96.5064% |
| Weighted F1 |     96.0257% |         97.5401% |

Based on these recorded values:

- Accuracy difference: **+1.5157 percentage points** for the hybrid.
- Macro F1 difference: **+2.1736 percentage points** for the hybrid.
- Weighted F1 difference: **+1.5144 percentage points** for the hybrid.

These are preliminary comparisons. Before making a final scientific claim that the hybrid approach is better, verify that both models used the same test split, label mapping, and comparable image preprocessing. Also verify the identity and provenance of the baseline checkpoint used for each reported metric.

Do not use the test set to choose additional hyperparameters. Further model-development decisions should be made using training and validation data.

## 12. Per-Class Error Analysis

The hybrid model's earlier classification report identified several difficult classes.

### Classes with lower F1 scores

| Class                     | Precision | Recall |     F1 | Test support |
| ------------------------- | --------: | -----: | -----: | -----------: |
| Corn Gray Leaf Spot       |    85.28% | 81.74% | 83.47% |          241 |
| Tomato Early Blight       |    85.86% | 87.90% | 86.87% |          463 |
| Apple Cedar Apple Rust    |    86.75% | 95.62% | 90.97% |          137 |
| Corn Northern Leaf Blight |    89.53% | 92.57% | 91.02% |          471 |
| Tomato Leaf Mold          |    91.67% | 94.57% | 93.10% |          442 |

### Frequently confused classes

The recorded test analysis showed confusion between visually similar disease categories.

| Actual class              | Predicted class           | Misclassifications |
| ------------------------- | ------------------------- | -----------------: |
| Corn Gray Leaf Spot       | Corn Northern Leaf Blight |                 36 |
| Corn Northern Leaf Blight | Corn Gray Leaf Spot       |                 24 |
| Tomato Late Blight        | Tomato Early Blight       |                 27 |
| Tomato Early Blight       | Tomato Late Blight        |                 13 |
| Tomato Septoria Leaf Spot | Tomato Early Blight       |                 17 |
| Tomato Spider Mites       | Tomato Target Spot        |                 15 |

Corn Gray Leaf Spot and Corn Northern Leaf Blight were frequently confused in both directions. Tomato Early Blight and Tomato Late Blight also showed notable confusion.

These results suggest that future improvements should focus on distinguishing similar symptoms, while preserving performance across the other classes.

**Evaluation note:** The class-level values above come from the earlier hybrid test report and should be used for diagnosis, not repeated test-driven model selection. Use validation reports to guide further experiments.

## 13. Evaluation Methodology

The project uses the following metrics:

| Metric           | Purpose                                     |
| ---------------- | ------------------------------------------- |
| Accuracy         | Overall fraction of correct predictions     |
| Macro precision  | Equal-weight average of per-class precision |
| Macro recall     | Equal-weight average of per-class recall    |
| Macro F1         | Equal-weight average of per-class F1        |
| Weighted F1      | F1 averaged according to class support      |
| Per-class report | Reveals weaker classes                      |
| Confusion matrix | Shows patterns of misclassification         |

The test set is reserved for final evaluation. Training and validation data should be used for training, checkpoint selection, and hyperparameter tuning.

For a fair baseline-versus-hybrid comparison, both pipelines must use the same test examples and consistent label mapping. Preprocessing differences must be documented.

## 14. Current Machine Learning Pipeline

```text
PlantVillage + PlantDoc
          |
          v
Dataset Cleaning and Validation
          |
          v
Canonical Class Mapping
          |
          v
Exact Duplicate Detection
          |
          v
Unified 38-Class Dataset
          |
          v
Stratified Train/Validation/Test Split
          |
          v
Image Preprocessing
          |
          v
Baseline CNN Training
          |
          v
CNN Checkpoint Selection
          |
          v
CNN Deep Feature Extraction
          |
          v
Feature Standardization
          |
          v
LinearSVC Training and Validation Tuning
          |
          v
Selected Hybrid Model
          |
          v
Test Evaluation and Error Analysis
          |
          v
Inference API
          |
          v
Frontend Integration
          |
          v
Deployment
```

## 15. Project Structure

The following is a documentation-oriented overview of relevant files. Verify the actual repository structure before assuming every path exists.

```text
plant-disease-hybrid/
|
|-- backend/
|   `-- ml_api/
|       |-- inference.py
|       `-- main.py
|
|-- configs/
|   `-- config.yaml
|
|-- data/
|   `-- processed/
|       |-- metadata/
|       `-- splits/
|
|-- outputs/
|   |-- baseline_cnn/
|   |-- baseline_cnn_full/
|   |-- baseline_cnn_retrained/
|   `-- hybrid_svm_run_20261029_125118/
|       |-- hybrid_svm.joblib
|       |-- run_summary.json
|       |-- test_metrics.json
|       |-- test_classification_report.csv
|       |-- test_confusion_matrix.csv
|       |-- train_features.npz
|       |-- val_features.npz
|       |-- test_features.npz
|       `-- svm_tuning_20261029_142949/
|
|-- scripts/
|   |-- analyze_hybrid_results.py
|   `-- tune_hybrid_svm.py
|
|-- src/
|   |-- data/
|   `-- models/
|
|-- .gitignore
`-- README.md
```

## 16. Reproducibility and GitHub

Avoid committing large datasets, virtual environments, and model binaries unless explicitly required by the project.

A recommended `.gitignore` configuration is:

```gitignore
# Virtual environments
.venv/
venv/

# Python cache
__pycache__/
*.py[cod]

# Secrets
.env

# Raw datasets
data/raw/

# Model checkpoints
*.pth
*.pt
*.ckpt
```

Adding a pattern to `.gitignore` does not remove files that are already tracked by Git.

Record the following for reproducibility:

- Python, PyTorch, torchvision, NumPy, pandas, and scikit-learn versions.
- Hardware and CUDA availability for each run.
- Dataset version and split definitions.
- Model architecture and checkpoint identity.
- Training configuration and random seed.
- Training history and checkpoint-selection criterion.
- SVM configuration and selected hyperparameters.
- Final test metrics and class-level reports.
- Image preprocessing and normalization configuration.

## 17. Next Development Milestones

### Milestone 1: Verify baseline checkpoint provenance

- Confirm which checkpoint corresponds to the supplied 20-epoch history.
- Verify the checkpoint-selection criterion and selected epoch.
- Preserve original model weights and logs.

### Milestone 2: Verify baseline test evaluation

- Confirm the source of the reported baseline test metrics.
- Verify the split, class mapping, and preprocessing.
- Save the final baseline classification report and confusion matrix.

### Milestone 3: Analyze difficult classes

- Use validation data to investigate Corn Gray Leaf Spot and Corn Northern Leaf Blight.
- Investigate confusion between Tomato Early Blight and Tomato Late Blight.
- Identify suitable improvements without using test results for model selection.

### Milestone 4: Improve the hybrid model

- Explore feature-extraction and training strategies.
- Compare changes using validation macro F1 and accuracy.
- Save each experiment in a separate output directory.
- Evaluate the selected final configuration on the reserved test set.

### Milestone 5: Application integration

- Complete model loading and inference.
- Integrate image preprocessing into the API.
- Return predicted class and suitable confidence information.
- Connect the frontend.
- Test the end-to-end image-prediction workflow.
- Prepare deployment and final project documentation.

## Conclusion

The project has completed dataset preparation, baseline CNN training for 20 recorded epochs, CNN feature extraction, and an initial hybrid CNN + SVM evaluation.

The baseline achieved its best recorded validation accuracy of **96.3607% at Epoch 17**. The hybrid classifier using `LinearSVC(C=1.0)` achieved **97.5361% test accuracy** and **96.5064% test macro F1** in the recorded experiment.

The SVM tuning experiment compared `C = 0.1`, `C = 1.0`, and `C = 10.0`. The best validation macro F1 was obtained at `C = 1.0`, so the original SVM configuration remains selected.

The next priorities are to verify checkpoint provenance and preprocessing consistency, investigate the difficult disease classes using validation data, and prepare the inference API and frontend integration.

**Current milestone:** Baseline CNN training and initial hybrid classification completed.

**Next milestone:** Reproducibility verification, validation-based error analysis, and application integration.
