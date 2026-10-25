# Plant Disease Hybrid Classification

A hybrid plant disease classification project combining the **PlantVillage** and **PlantDoc** datasets to develop a multi-class plant disease recognition system. PlantVillage provides controlled plant-leaf imagery, while PlantDoc contributes images captured under more diverse real-world conditions.

The project uses PyTorch to train a baseline Convolutional Neural Network (CNN) and is designed to compare its performance with a hybrid classification approach.

## Current Project Status

| Component | Status |
|---|---|
| PlantVillage dataset collection | Completed |
| PlantDoc dataset collection and extraction | Completed |
| Dataset class mapping | Completed |
| Unified dataset construction | Completed |
| Missing-file validation | Completed |
| Exact duplicate detection | Completed |
| Stratified train/validation/test splitting | Completed |
| Dataset and DataLoader verification | Completed |
| Image preprocessing and augmentation tests | Completed |
| CUDA GPU verification | Completed |
| Baseline CNN implementation | Completed |
| Smoke-test training pipeline | Completed |
| Full-dataset baseline CNN training | 10 epochs completed |
| Baseline training history | Available |
| Best baseline checkpoint | Saved |
| Full-dataset test evaluation | Pending |
| Hybrid model implementation | Pending |
| Deep feature extraction | Pending |
| Baseline-versus-hybrid comparison | Pending |
| Inference API | Pending |
| Frontend integration | Planned |
| Deployment | Planned |

## 1. Project Overview

The objective is to classify plant leaf images into 38 canonical plant health and disease categories.

The project workflow includes:

1. Collecting PlantVillage and PlantDoc images.
2. Mapping source labels to a unified class taxonomy.
3. Validating image paths and removing exact duplicates.
4. Creating stratified training, validation, and test splits.
5. Preprocessing images and verifying PyTorch DataLoaders.
6. Training a baseline CNN.
7. Evaluating the selected baseline model on the reserved test set.
8. Developing and evaluating the hybrid classification approach.
9. Comparing both models using consistent evaluation metrics.
10. Integrating the final model into an inference API and frontend.

## 2. Unified Dataset

The prepared dataset contains **164,501 images across 38 classes**.

| Dataset component | Number of images |
|---|---:|
| PlantVillage | 162,853 |
| PlantDoc | 1,648 |
| **Total** | **164,501** |
| Canonical classes | 38 |
| Missing image files detected | 0 |
| Exact duplicate images removed | 63 |

PlantVillage supplies most of the images, while PlantDoc adds greater variation in backgrounds and image-capture conditions.

Only source classes that could be reliably mapped to the project's canonical class taxonomy were included.

### Dataset split

A stratified 70/15/15 split was prepared.

| Split | Number of images | Number of classes |
|---|---:|---:|
| Training | 115,150 | 38 |
| Validation | 24,675 | 38 |
| Test | 24,676 | 38 |
| **Total** | **164,501** | **38** |

All 38 classes are represented in each split. The test split is reserved for final evaluation and must not be used for training or model selection.

## 3. Canonical Classes

The dataset contains the following 38 classes:

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

Contains the unified dataset metadata, including image paths, source dataset, original class labels, canonical labels, and class IDs.

Important columns include:

- `image_path`
- `source_dataset`
- `raw_class`
- `label`
- `class_id`

### Class mapping

```text
data/processed/metadata/class_mapping.csv
```

Contains the mapping between source labels and canonical class names.

### Dataset statistics

```text
data/processed/metadata/hybrid_dataset_statistics.csv
```

Contains dataset and class statistics.

### Duplicate report

```text
data/processed/metadata/unified_duplicates.csv
```

Contains information about detected duplicate images.

### Data splits

```text
data/processed/splits/train.csv
data/processed/splits/val.csv
data/processed/splits/test.csv
```

These files define the training, validation, and test datasets.

## 5. Development Environment

The project was developed on Windows using Visual Studio Code and PowerShell.

| Component | Configuration |
|---|---|
| Operating system | Windows |
| Python | 3.14.5 |
| PyTorch | 2.14.1+cu126 |
| Torchvision | 0.29.1+cu126 |
| GPU | NVIDIA GeForce 930MX |
| CUDA availability | Verified |
| Input resolution | 224 × 224 |
| Number of classes | 38 |
| Optimizer | Adam |
| Loss function | Cross-entropy loss |
| Learning rate | 0.001 |
| Batch size | 4 |

**Hardware note:** Training results in this README were recorded on the NVIDIA GeForce 930MX. Results may differ on other hardware.

The Python virtual environment is local to the development machine and should not be committed to GitHub.

## 6. Dataset and Preprocessing Verification

Before training, the dataset loading and preprocessing pipeline was tested.

The verification confirmed that:

- All split CSV files load correctly.
- No missing image files were detected during dataset validation.
- All 38 classes are represented in each split.
- Images and labels load correctly through PyTorch DataLoaders.
- Input image tensors have shape `(3, 224, 224)`.
- Image tensors use `torch.float32`.
- Label tensors use `torch.int64`.
- Tensor values are finite.
- Training and evaluation transformations execute successfully.
- Batches can be transferred to the CUDA device.
- Forward and backward passes execute successfully.

**Status: Dataset loading, preprocessing, and model execution checks passed.**

## 7. Baseline CNN Architecture

The baseline model is a custom convolutional neural network implemented in PyTorch.

### Architecture overview

The model contains four convolutional stages:

1. Convolutional layer with 32 output channels.
2. Convolutional layer with 64 output channels.
3. Convolutional layer with 128 output channels.
4. Convolutional layer with 256 output channels.

The architecture uses batch normalization and ReLU activation. Max pooling is applied in the earlier stages to reduce spatial dimensions.

Adaptive average pooling converts the final feature map into a fixed-size representation.

The classification head consists of:

- Flattening
- Dropout with probability 0.3
- A fully connected layer producing 38 class scores

The model contains approximately **399,142 trainable parameters**.

### Training configuration

| Parameter | Value |
|---|---|
| Model | BaselineCNN |
| Training images | 115,150 |
| Validation images | 24,675 |
| Planned training duration | 20 epochs |
| Epochs completed | 10 |
| Batch size | 4 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | Cross-entropy loss |
| Input resolution | 224 × 224 |
| Device | NVIDIA GeForce 930MX |

## 8. Smoke-Test Training

Before full-dataset training, a small smoke test was conducted to verify the training pipeline.

| Parameter | Value |
|---|---:|
| Training subset | 1,000 images |
| Validation subset | 200 images |
| Epochs | 3 |
| Batch size | 4 |
| Learning rate | 0.001 |
| Best validation accuracy | 24.00% |
| Test accuracy in smoke test | 24.23% |

These results belong to the preliminary smoke test and are not the final benchmark for the full-dataset model.

The smoke-test artifacts are stored in:

```text
outputs/baseline_cnn/
```

## 9. Full-Dataset Baseline CNN Training

The baseline CNN was trained using all 115,150 training images and evaluated against all 24,675 validation images during each recorded epoch.

Training history is stored in:

```text
outputs/baseline_cnn_full/training_history.csv
```

### Complete training results: Epochs 1–10

The following results are taken from the recorded training history.

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Duration (seconds) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.714159 | 51.35% | 0.781721 | 77.42% | 6121.67 |
| 2 | 0.817357 | 75.44% | 0.503964 | 86.49% | 6141.33 |
| 3 | 0.571830 | 82.69% | 0.417845 | 88.77% | 6129.22 |
| 4 | 0.452984 | 86.29% | 0.371728 | 90.05% | 6092.54 |
| 5 | 0.379222 | 88.53% | 0.327342 | 91.38% | 6076.53 |
| 6 | 0.326542 | 90.20% | 0.309455 | 92.11% | 6097.88 |
| 7 | 0.286395 | 91.39% | 0.300899 | 92.54% | 6028.05 |
| 8 | 0.255193 | 92.39% | **0.297089** | 92.74% | 6083.22 |
| 9 | 0.228878 | 93.11% | 0.304231 | 92.83% | 6053.57 |
| 10 | **0.207024** | **93.81%** | 0.307952 | **93.05%** | 6115.12 |

*Accuracy values are converted from the recorded decimal values to percentages. Training durations are rounded to two decimal places.*

### Training progress summary

| Metric | Epoch 1 | Epoch 10 |
|---|---:|---:|
| Training loss | 1.714159 | 0.207024 |
| Training accuracy | 51.35% | 93.81% |
| Validation loss | 0.781721 | 0.307952 |
| Validation accuracy | 77.42% | 93.05% |

The training loss decreased from 1.714159 to 0.207024 over the first 10 epochs. Training accuracy increased from 51.35% to 93.81%.

Validation accuracy improved from 77.42% in Epoch 1 to 93.05% in Epoch 10.

### Best recorded validation metrics

- **Highest validation accuracy:** 93.05%, achieved in Epoch 10.
- **Lowest validation loss:** 0.297089, achieved in Epoch 8.
- **Training accuracy at Epoch 10:** 93.81%.
- **Training loss at Epoch 10:** 0.207024.

The lowest validation loss and highest validation accuracy occurred in different epochs. This is possible because loss and accuracy measure different aspects of model performance.

Validation loss increased slightly in Epochs 9 and 10 even as validation accuracy improved. This should be monitored during further training because it may indicate emerging overfitting or changes in prediction confidence.

### Training time

Each epoch took approximately 100–103 minutes on the recorded hardware.

The total recorded training time across the first 10 epochs was approximately **16.91 hours**.

### Training output directory

```text
outputs/baseline_cnn_full/
```

The directory contains the baseline training artifacts, including the training history, training curves, and saved model checkpoint, depending on the outputs produced by the training scripts.

### Checkpoint

The best model checkpoint is saved at:

```text
outputs/baseline_cnn_full/best_model.pth
```

The training script uses validation performance to select the best checkpoint. Confirm the saved checkpoint's epoch and selection criterion before using it for final evaluation.

**Important:** The reported 93.05% is validation accuracy, not test accuracy. Final performance must be measured on the reserved test set.

## 10. Training Curves

The training curves can be inspected at:

```text
outputs/baseline_cnn_full/training_curves.png
```

The plots should be used to observe:

- Changes in training loss over the epochs.
- Changes in validation loss over the epochs.
- Improvements in training accuracy.
- Improvements in validation accuracy.
- Possible divergence between training and validation performance.

The recorded history shows consistent improvement in training accuracy. Validation accuracy also improves across the 10 recorded epochs, while validation loss reaches its minimum in Epoch 8 and increases slightly afterward.

## 11. Current Model Evaluation Status

The full-dataset baseline has completed 10 recorded training epochs. The following tasks remain pending:

- Final evaluation on the reserved test split.
- Overall test accuracy.
- Macro precision, recall, and F1-score.
- Weighted precision, recall, and F1-score.
- Per-class classification report.
- Confusion matrix.
- Error analysis.
- Final comparison against the hybrid model.

The reserved test set must remain separate from model training and checkpoint selection. Use the training and validation splits for development decisions, then evaluate the selected model on the test split.

## 12. Planned Hybrid Classification Approach

The project's next major objective is to develop and evaluate the hybrid classification model.

The exact hybrid architecture should be finalized before implementation. One possible approach is to use a trained CNN as a feature extractor and train a downstream classifier, such as a Support Vector Machine (SVM), using the extracted deep features.

If the CNN + SVM approach is selected, the workflow will be:

1. Load the chosen trained CNN checkpoint.
2. Remove or bypass its final classification layer to obtain deep feature vectors.
3. Extract features for the training and validation images.
4. Train the SVM using training features and labels.
5. Select the downstream classifier configuration using validation performance.
6. Evaluate the final hybrid pipeline on the reserved test set.
7. Compare its performance with the baseline CNN.

The selected CNN feature extractor should remain fixed while extracting features for a given SVM experiment. If the CNN checkpoint changes, the features should be extracted again for a fair and consistent experiment.

## 13. Model Comparison Plan

The baseline and hybrid approaches should be evaluated on the same held-out test split.

The planned comparison includes:

| Evaluation metric | Purpose |
|---|---|
| Accuracy | Measures overall classification correctness |
| Macro precision | Measures precision averaged equally across classes |
| Macro recall | Measures recall averaged equally across classes |
| Macro F1-score | Measures class-balanced F1 performance |
| Weighted F1-score | Accounts for class support |
| Per-class metrics | Identifies classes with weaker performance |
| Confusion matrix | Shows patterns of misclassification |
| Inference time | Compares prediction speed |
| Computational requirements | Documents resource usage |

Final results should be reported only after the corresponding evaluations have been completed. No final test score or hybrid-model performance is claimed at this stage.

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
Dataset and Preprocessing Verification
          |
          v
Baseline CNN Training
          |
          v
Validation and Checkpoint Selection
          |
          v
Baseline Test Evaluation
          |
          v
Hybrid Model Development
          |
          v
Feature Extraction / Hybrid Classification
          |
          v
Baseline-versus-Hybrid Comparison
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

The following is a documentation-oriented view of the relevant project files. Verify the actual directory contents before assuming every listed path exists.

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
|   |-- raw/
|   `-- processed/
|       |-- metadata/
|       |   |-- unified_dataset.csv
|       |   |-- class_mapping.csv
|       |   |-- hybrid_dataset_statistics.csv
|       |   `-- unified_duplicates.csv
|       `-- splits/
|           |-- train.csv
|           |-- val.csv
|           `-- test.csv
|
|-- evaluation/
|   `-- reports/
|
|-- models/
|   `-- cnn/
|
|-- outputs/
|   |-- baseline_cnn/
|   `-- baseline_cnn_full/
|       |-- training_history.csv
|       |-- training_curves.png
|       `-- best_model.pth
|
|-- scripts/
|   `-- inspect_dataset.py
|
|-- src/
|   |-- data/
|   |   |-- clean_dataset.py
|   |   `-- prepare_splits.py
|   `-- models/
|       |-- cnn_backbones.py
|       |-- extract_deep_features.py
|       `-- train_cnn.py
|
|-- baseline_cnn.py
|-- dataset_loader.py
|-- test_preprocessing.py
|-- check_model.py
|-- train_baseline.py
|-- train_baseline_full.py
|-- continue_baseline.py
|-- .gitignore
`-- README.md
```

## 16. Reproducibility and GitHub

Avoid committing large datasets, virtual environments, or model binaries unless the project explicitly requires them.

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

If these files have already been committed, adding them to `.gitignore` alone will not remove them from Git tracking.

When moving the project to another computer, install the required dependencies in a new virtual environment and update any dataset paths that point to machine-specific locations.

Record the following for reproducibility:

- Python and library versions.
- GPU model and CUDA availability.
- Dataset version and split definitions.
- Model architecture and training configuration.
- Random seed.
- Training history and checkpoint selection criterion.
- Final test results.
- Hybrid-model configuration and evaluation results.

## 17. Next Development Milestones

### Milestone 1: Baseline training review

- Review all 10 recorded epochs.
- Inspect `training_history.csv` and `training_curves.png`.
- Confirm the epoch associated with `best_model.pth`.
- Decide whether to continue training toward the planned 20 epochs.
- Preserve existing checkpoints and logs.

### Milestone 2: Baseline test evaluation

- Load the selected CNN checkpoint.
- Evaluate the model on the reserved test split.
- Generate accuracy, precision, recall, and F1-score metrics.
- Save the per-class report and confusion matrix.
- Document test results separately from validation results.

### Milestone 3: Hybrid model development

- Finalize the hybrid architecture.
- Implement the selected feature-extraction or hybrid-classification pipeline.
- Save separate checkpoints and experiment logs.
- Tune model choices using validation data only.

### Milestone 4: Performance comparison

- Evaluate the selected hybrid model on the same test split.
- Compare the baseline and hybrid metrics.
- Analyze per-class errors and confusion patterns.
- Record inference time and computational requirements.

### Milestone 5: Application integration

- Complete the inference pipeline.
- Integrate image preprocessing and model loading into the API.
- Return predicted classes and appropriate confidence information.
- Connect the frontend.
- Test the complete image-prediction workflow.
- Prepare deployment and final project documentation.

## Conclusion

The dataset preparation and validation stages are complete, and the baseline CNN has completed **10 full-dataset training epochs**.

At Epoch 10, the model achieved **93.81% training accuracy and 93.05% validation accuracy**. The lowest recorded validation loss was **0.297089 at Epoch 8**.

The next milestones are to review the saved checkpoint, complete the baseline's held-out test evaluation, and implement the hybrid classification approach. A final test score and a baseline-versus-hybrid comparison have not yet been established.

**Current milestone:** Baseline CNN training — 10 epochs completed.

**Next milestone:** Checkpoint verification, baseline test evaluation, and hybrid model development.
*
