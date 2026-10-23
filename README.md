# Plant Disease Hybrid Classification

A hybrid plant disease classification project combining **PlantVillage** and **PlantDoc** datasets to develop a multi-class plant disease recognition system. PlantVillage provides controlled plant-leaf imagery, while PlantDoc adds real-world images with more diverse backgrounds and imaging conditions.

The project uses PyTorch for model development and is designed to compare a baseline CNN with a hybrid classification approach.

## Current Project Status

| Component | Status |
|---|---|
| PlantVillage collection | ✅ Completed |
| PlantDoc collection and extraction | ✅ Completed |
| Dataset class mapping | ✅ Completed |
| Unified dataset construction | ✅ Completed |
| Missing-file validation | ✅ Completed |
| Exact duplicate detection | ✅ Completed |
| Stratified train/validation/test splitting | ✅ Completed |
| Dataset and DataLoader verification | ✅ Completed |
| Image preprocessing and augmentation tests | ✅ Completed |
| CUDA GPU verification | ✅ Completed |
| Baseline CNN implementation | ✅ Completed |
| Smoke-test training pipeline | ✅ Completed |
| Full-dataset Baseline CNN training | 🟢 Two epochs completed |
| Best baseline checkpoint | ✅ Saved |
| Final baseline test evaluation | ⏳ Pending |
| Hybrid model implementation | ⏳ Next |
| Deep feature extraction | ⏳ Pending |
| Model comparison | ⏳ Pending |
| Inference API | ⏳ Under development / pending |
| Frontend | ⏳ Planned |
| Deployment | ⏳ Planned |

## 1. Final Hybrid Dataset

The unified dataset contains **164,501 images across 38 classes**.

| Dataset component | Count |
|---|---:|
| PlantVillage images | 162,853 |
| PlantDoc images | 1,648 |
| Total images | 164,501 |
| Canonical classes | 38 |
| Missing image files | 0 |
| Exact duplicate images removed | 63 |

Only PlantDoc classes that could be mapped reliably to the project's canonical class taxonomy were included.

### Dataset split

A stratified 70/15/15 split was created.

| Split | Images | Classes |
|---|---:|---:|
| Training | 115,150 | 38 |
| Validation | 24,675 | 38 |
| Test | 24,676 | 38 |
| **Total** | **164,501** | **38** |

All 38 classes are represented in each split, and dataset validation confirmed that all referenced image files exist.

## 2. Dataset Sources

### PlantVillage

PlantVillage supplies the majority of the images, covering multiple crops and plant disease categories. Its controlled imagery provides a useful foundation for learning disease-related visual patterns.

### PlantDoc

PlantDoc contributes images captured under more varied real-world conditions. Its inclusion is intended to help investigate model performance beyond controlled backgrounds.

The final dataset combines these sources using a unified class mapping.

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

Contains unified metadata for the dataset.

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

Contains the mapping from source labels to canonical classes.

### Dataset statistics

```text
data/processed/metadata/hybrid_dataset_statistics.csv
```

Contains dataset and class statistics.

### Duplicate report

```text
data/processed/metadata/unified_duplicates.csv
```

Contains the results of duplicate-image detection.

### Final data splits

```text
data/processed/splits/train.csv
data/processed/splits/val.csv
data/processed/splits/test.csv
```

These files define the final training, validation, and test splits.

## 5. Development Environment

The project has been developed in Windows using Visual Studio Code and PowerShell.

| Component | Verified configuration |
|---|---|
| Python | 3.14.5 |
| PyTorch | 2.14.1+cu126 |
| Torchvision | 0.29.1+cu126 |
| GPU | NVIDIA GeForce 930MX |
| CUDA availability | Verified |
| Input resolution | 224 × 224 |
| Number of classes | 38 |
| Optimizer | Adam |
| Loss function | Cross-entropy loss |

**Hardware note:** The full-dataset training results documented below were produced on the NVIDIA GeForce 930MX. If training is moved to another machine, record that machine's hardware and software versions separately.

### Virtual environment

The project uses a Python virtual environment. The environment directory should not be committed to Git.

## 6. Dataset and Preprocessing Verification

The dataset pipeline was verified before model training.

The checks confirmed:

- All split CSV files load correctly.
- No missing image files were detected.
- All 38 classes are present in each split.
- Images and labels load correctly through PyTorch DataLoaders.
- Image tensors have shape `(3, 224, 224)`.
- Image tensors use `torch.float32`.
- Label tensors use `torch.int64`.
- Tensor values are finite.
- Training and evaluation transformations execute successfully.
- Batches can be transferred to the CUDA device.

**Status: Dataset loading, preprocessing, and augmentation tests passed.**

## 7. Baseline CNN Architecture

A custom convolutional neural network named `BaselineCNN` has been implemented as the initial classification model.

### Architecture

The model contains four convolutional stages:

1. Convolution with 32 output channels
2. Convolution with 64 output channels
3. Convolution with 128 output channels
4. Convolution with 256 output channels

The convolutional stages use batch normalization and ReLU activation, with max pooling in the earlier stages. Adaptive average pooling reduces the final feature map to a fixed-size representation.

The classification head contains:

- Flattening
- Dropout with probability 0.3
- A fully connected layer producing 38 class scores

The model contains approximately **399,142 parameters**.

A forward pass, loss calculation, and backward pass were successfully verified on the CUDA device.

### Relevant files

```text
baseline_cnn.py
check_model.py
dataset_loader.py
test_preprocessing.py
train_baseline.py
train_baseline_full.py
continue_baseline.py
```

The scripts cover the baseline architecture, dataset loading, preprocessing verification, smoke testing, full-dataset training, and checkpoint-based continuation. The continuation script is prepared and has completed one continuation run.

## 8. Smoke-Test Training

Before full-dataset training, the training pipeline was tested on a smaller subset to verify that training, validation, checkpoint saving, and report generation worked end to end.

| Parameter | Value |
|---|---:|
| Training subset | 1,000 images |
| Validation subset | 200 images |
| Epochs | 3 |
| Batch size | 4 |
| Learning rate | 0.001 |
| Best validation accuracy | 24.00% |
| Test accuracy | 24.23% |

These results are from the smoke test only. They are not the final benchmark because the model was trained on a small subset.

The smoke-test artifacts are stored in:

```text
outputs/baseline_cnn/
```

The test results from this preliminary run must not be presented as the final test performance of the full-dataset model.

## 9. Full-Dataset Baseline Training

The Baseline CNN was trained using all 115,150 training images and validated using all 24,675 validation images.

### Training configuration

| Parameter | Value |
|---|---|
| Model | BaselineCNN |
| Training images | 115,150 |
| Validation images | 24,675 |
| Completed epochs | 2 |
| Batch size | 4 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | Cross-entropy loss |
| Input resolution | 224 × 224 |
| Device | NVIDIA GeForce 930MX |

### Training history

| Metric | Epoch 1 | Epoch 2 |
|---|---:|---:|
| Training loss | 1.7142 | 0.8174 |
| Training accuracy | 51.35% | 75.44% |
| Validation loss | 0.7817 | 0.5040 |
| Validation accuracy | 77.42% | 86.49% |
| Epoch duration | 6,121.7 s | 6,141.3 s |

### Current baseline result

**Best validation accuracy: 86.49%**

The second epoch improved validation accuracy by approximately 9.07 percentage points. Both training and validation losses decreased between the two epochs.

These results indicate encouraging progress, but further validation monitoring is needed before determining whether more training is beneficial.

**Important:** The 86.49% figure is validation accuracy, not test accuracy.

### Training output directory

```text
outputs/baseline_cnn_full/
```

The directory contains the best model checkpoint and the training history and curve artifacts generated by the training scripts.

### Checkpoint details

The best checkpoint is saved at:

```text
outputs/baseline_cnn_full/best_model.pth
```

The checkpoint contains:

- Model state dictionary
- Optimizer state dictionary
- Best validation accuracy
- Epoch number
- Number of classes
- Class names
- Random seed

The continuation run loaded the existing model and optimizer states and saved an updated best checkpoint after validation accuracy improved.

## 10. Current Model Evaluation Status

The following evaluation steps remain pending for the full-dataset baseline:

- Final evaluation on the reserved test split
- Overall test accuracy
- Macro precision, recall, and F1-score
- Weighted precision, recall, and F1-score
- Per-class classification metrics
- Confusion matrix
- Error analysis
- Final comparison with the hybrid model

The test set was intentionally excluded from the full-dataset training and continuation runs.

Model selection and training decisions should be based on the training and validation splits. The reserved test set should be used for final evaluation after model selection.

## 11. Current ML Pipeline

The current workflow is:

```text
PlantVillage + PlantDoc
          ↓
Dataset Cleaning
          ↓
Class Mapping
          ↓
Duplicate Detection
          ↓
Unified 38-Class Dataset
          ↓
Stratified 70/15/15 Split
          ↓
Dataset and Preprocessing Verification
          ↓
Baseline CNN Training
          ↓
Validation and Checkpoint Selection
          ↓
Final Baseline Test Evaluation
          ↓
Hybrid Architecture Development
          ↓
Deep Feature Extraction / Hybrid Classification
          ↓
Model Comparison and Error Analysis
          ↓
Inference API
          ↓
Frontend Integration
          ↓
Deployment
```

## 12. Project Structure

The relevant project structure is:

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
│   └── processed/
│       ├── metadata/
│       │   ├── unified_dataset.csv
│       │   ├── class_mapping.csv
│       │   ├── hybrid_dataset_statistics.csv
│       │   └── unified_duplicates.csv
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
├── outputs/
│   ├── baseline_cnn/
│   └── baseline_cnn_full/
│
├── scripts/
│   └── inspect_dataset.py
│
├── src/
│   ├── data/
│   │   ├── clean_dataset.py
│   │   └── prepare_splits.py
│   └── models/
│       ├── cnn_backbones.py
│       ├── extract_deep_features.py
│       └── train_cnn.py
│
├── baseline_cnn.py
├── dataset_loader.py
├── test_preprocessing.py
├── check_model.py
├── train_baseline.py
├── train_baseline_full.py
├── continue_baseline.py
├── .gitignore
└── README.md
```

This is a documentation view of the project structure; verify the actual directory contents before assuming every listed path exists in the repository.

## 13. Next Development Tasks

### Task 1: Review baseline training

- Inspect `training_history.csv` and `training_curves.png`.
- Decide whether another training epoch is worthwhile.
- Preserve the best checkpoint.
- Avoid restarting training from scratch unnecessarily.

### Task 2: Complete baseline evaluation

- Load the selected checkpoint.
- Evaluate it on the reserved test split.
- Generate overall and per-class metrics.
- Save the classification report and confusion matrix.
- Document test results separately from validation results.

### Task 3: Implement the hybrid model

The project is intended to compare the baseline CNN against a hybrid classification approach.

Before implementation, finalize the architecture. Depending on the project's approved design, this may combine a CNN backbone with an attention mechanism, a Transformer, or a feature-based classifier.

The hybrid model should:

- Use the same 38-class taxonomy.
- Use the established training and validation splits.
- Save its own checkpoints and logs.
- Be evaluated using the same metrics as the baseline.
- Avoid overwriting the baseline artifacts.

### Task 4: Deep feature extraction

If required by the chosen hybrid architecture:

- Extract deep feature vectors using the trained backbone.
- Save features and their corresponding labels.
- Train the selected downstream classifier using training features.
- Use validation features for model selection.
- Reserve test features for final evaluation.

### Task 5: Compare model performance

Compare the baseline and hybrid approaches using:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score
- Per-class performance
- Confusion matrices
- Inference time and computational requirements

Use consistent data splits and evaluation procedures for a fair comparison.

### Task 6: API and application integration

After the model and inference pipeline are stable:

- Integrate model loading and preprocessing into the inference API.
- Validate uploaded image handling.
- Return the predicted class and relevant confidence scores.
- Connect the frontend.
- Test the complete prediction workflow.
- Prepare deployment documentation.

## 14. Git and Reproducibility Rules

Do not commit large datasets, virtual environments, or generated model binaries unless the project explicitly requires them.

Recommended exclusions include:

```gitignore
data/raw/
.venv/
__pycache__/
*.pyc
*.pth
*.pt
*.ckpt
```

The repository should contain the code, configuration, dataset metadata, split definitions, documentation, and relevant reports required to reproduce the pipeline.

Keep a record of:

- Python and library versions
- GPU model
- Dataset split definitions
- Training configuration
- Random seed
- Model checkpoints and validation results
- Final test results

Because the CSV files may contain local absolute paths, verify or configure dataset paths when cloning the project onto another machine.

## 15. Development Milestones

### Completed

- PlantVillage and PlantDoc integration
- Unified 38-class dataset construction
- Missing-file validation
- Exact duplicate detection
- Stratified train/validation/test splits
- Dataset and DataLoader verification
- Preprocessing and augmentation tests
- CUDA verification
- Baseline CNN implementation
- Smoke-test pipeline
- Two full-dataset baseline training epochs
- Checkpoint-based continuation
- Best baseline validation accuracy of 86.49%

### In progress / next

- Baseline training review
- Final baseline test evaluation
- Hybrid architecture implementation
- Deep feature extraction, if required
- Hybrid model training
- Baseline-versus-hybrid comparison
- Detailed evaluation and error analysis

### Planned

- Inference API completion
- Frontend integration
- End-to-end application testing
- Deployment
- Final project report

## Conclusion

The dataset preparation phase is complete, and the first full-dataset Baseline CNN has been successfully trained for two epochs. It currently achieves **86.49% validation accuracy**, with its best checkpoint saved under `outputs/baseline_cnn_full/`.

The next milestone is to review the baseline learning curves, perform final test evaluation after model selection, and develop the hybrid classification architecture. The final test performance and baseline-versus-hybrid comparison have not yet been established.

**Current milestone: Baseline CNN training — two epochs completed.**

**Next milestone: Final baseline evaluation and hybrid model development.**
