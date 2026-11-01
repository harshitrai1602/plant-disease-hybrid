# Plant Disease Hybrid Classification

A multi-class plant disease recognition project combining the
**PlantVillage** and **PlantDoc** datasets. The project uses a PyTorch
Convolutional Neural Network (CNN) as a baseline and a hybrid **CNN
feature extractor + Support Vector Machine (SVM)** classifier for
comparison.

The unified dataset is reported as containing **164,501 images across 38
canonical plant health and disease classes**.

> **Project status note:** This README combines the supplied project
> README with the later *Baseline Validation Report*. Where the records
> contain conflicting
> checkpoint paths, metric availability, or environment details, those
> points are explicitly marked for verification rather than silently
> treated as settled facts.

------------------------------------------------------------------------

## 1. Current Project Status

The following table summarises the development progress of the Plant Disease Hybrid Classification project.

| Component | Status |
|---|---|
| PlantVillage and PlantDoc dataset collection | ✅ Completed |
| Dataset class mapping and unification | ✅ Completed |
| Missing-file validation | ✅ Completed |
| Exact duplicate detection | ✅ Completed |
| Stratified train/validation/test splitting | ✅ Completed |
| Dataset and DataLoader verification | ✅ Completed |
| Image preprocessing and augmentation tests | ✅ Completed |
| Baseline CNN training | ✅ 20 epochs recorded |
| Baseline checkpoint | ✅ Saved (provenance verification pending) |
| Baseline validation evaluation | ✅ Completed |
| Baseline held-out test evaluation | ✅ Completed |
| Hybrid CNN feature extraction | ✅ Completed |
| Initial CNN + SVM model | ✅ Completed |
| SVM hyperparameter comparison | ✅ Completed |
| Hybrid validation evaluation | ✅ Completed |
| Hybrid test evaluation | ✅ Completed |
| Baseline-versus-hybrid validation comparison | ✅ Completed |
| Baseline-versus-hybrid test comparison | ✅ Completed |
| Per-class performance and confusion analysis | ✅ Completed |
| Misclassification analysis | ✅ Completed |
| Inference API | ⏳ Pending |
| Frontend integration | Not planned |
| Deployment | Not planned |

### Notes on Evaluation

- Baseline and hybrid models have been evaluated on both validation and held-out test datasets.
- Comparative evaluation includes overall metrics, per-class performance, confusion matrices, and misclassification analysis.
- Validation comparisons are structurally aligned. Preprocessing and feature provenance should remain documented.
- The baseline checkpoint has been saved; its exact provenance still requires verification.

### Execution Support

The project supports both CPU and GPU-based execution:

- **CPU:** Runs on systems without a dedicated GPU.
- **GPU + CUDA:** Uses a compatible NVIDIA GPU when CUDA-enabled PyTorch and the required drivers are available.

The execution device is selected according to the available hardware and software environment.
## 2. Project Overview

The objective is to classify plant leaf images into 38 canonical plant
health and disease categories.

PlantVillage contributes predominantly controlled-condition images,
while PlantDoc introduces greater variation in backgrounds and
image-capture conditions.

The project workflow is:

1.  Collect and validate the datasets.
2.  Map source labels to a unified taxonomy.
3.  Detect and remove exact duplicate images.
4.  Create stratified train, validation, and test splits.
5.  Train and evaluate a baseline CNN.
6.  Extract deep features using the trained CNN.
7.  Standardize the extracted features.
8.  Train an SVM on the extracted training features.
9.  Select SVM hyperparameters using validation performance.
10. Evaluate the selected hybrid model on the reserved test split.
11. Compare model performance and prepare application integration.

### Hybrid model architecture

``` text
Plant Leaf Image
      |
      v
Image Preprocessing (224 x 224)
      |
      v
Trained Baseline CNN
      |
      v
256-dimensional deep feature vector
      |
      v
StandardScaler
      |
      v
LinearSVC
      |
      v
One of 38 canonical classes
```

The hybrid experiment reuses the saved CNN checkpoint. It does not
retrain the CNN during feature extraction and SVM evaluation.

## 3. Unified Dataset

The prepared dataset is reported as containing 164,501 images across 38
classes.

  Dataset component                                   Number of images
  ------------------------------------------------- ------------------
  PlantVillage                                                 162,853
  PlantDoc                                                       1,648
  Total reported images                                        164,501
  Canonical classes                                                 38
  Missing image files detected during preparation                    0
  Exact duplicate images removed                                    63

Only source classes that could be reliably mapped to the project's
canonical taxonomy were included.

**Dataset-statistics clarification:** The source README reports the
source counts, total, and duplicate-removal count together. Confirm
whether the source counts/total refer to pre-deduplication or
post-deduplication records before describing 164,501 as the final
post-deduplication image count in a formal thesis.

### Dataset split

A stratified 70/15/15 split was prepared.

  Split               Images   Classes
  ------------ ------------- ---------
  Training           115,150        38
  Validation          24,675        38
  Test                24,676        38
  **Total**      **164,501**    **38**

The test split is reserved for final evaluation. It must not be used for
training, hyperparameter selection, or model selection.

## 4. Canonical Classes

The 38 classes are:

``` text
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

Use the saved class mapping and checkpoint class ordering for inference
and evaluation. Do not rely on a newly sorted class list if it differs
from the saved mapping.

## 5. Important Dataset Files

### Unified metadata

``` text
data/processed/metadata/unified_dataset.csv
```

Important columns:

-   `image_path`
-   `source_dataset`
-   `raw_class`
-   `label`
-   `class_id`

### Other metadata

``` text
data/processed/metadata/class_mapping.csv
data/processed/metadata/hybrid_dataset_statistics.csv
data/processed/metadata/unified_duplicates.csv
```

These files contain label mappings, dataset statistics, and
duplicate-detection information.

### Data splits

``` text
data/processed/splits/train.csv
data/processed/splits/val.csv
data/processed/splits/test.csv
```

Older files may also exist:

``` text
data/processed/splits/train_old_wrong.csv
data/processed/splits/val_old_wrong.csv
data/processed/splits/test_old_wrong.csv
```

Use the corrected split files for experiments. Preserve the existing
split definitions when comparing models.

### Important image-path issue

Some CSV records contain paths from a teammate's previous computer,
beginning with:

``` text
D:\MAJOR\plant-disease-hybrid\
```

The current project directory is:

``` text
C:\Users\ottaa\OneDrive\Desktop\Minor Project\plant-disease-hybrid
```

Those old absolute paths may not exist on the current laptop. Before
loading images again, verify that every path resolves to an existing
image. Do not rebuild the dataset or regenerate splits unnecessarily.

## 6. Development Environment

The project uses Windows, Visual Studio Code, PowerShell, Python,
PyTorch, Torchvision, and scikit-learn.

  -----------------------------------------------------------------------
  Component                           Recorded configuration
  ----------------------------------- -----------------------------------
  Operating system                    Windows

  Python                              3.14.5, as recorded in project
                                      documentation

  Input resolution                    224 x 224

  Number of classes                   38

  Baseline optimizer                  Adam

  Baseline loss                       Cross-entropy

  Baseline learning rate              0.001 in the supplied 20-epoch
                                      training record

  Baseline batch size                 4 in the supplied 20-epoch training
                                      record

  Hybrid classifier                   LinearSVC

  Hybrid feature scaling              StandardScaler

  Hybrid feature dimension            256
  -----------------------------------------------------------------------

### CPU and GPU execution support

The project is designed to run on either **CPU** or **GPU with CUDA**,
depending on the processing hardware available on the machine and the
installed PyTorch build:

- **CPU mode:** Runs on systems with a supported CPU, even when no
  compatible GPU is present.
- **GPU + CUDA mode:** Can use an NVIDIA GPU when the hardware, driver,
  CUDA-enabled PyTorch installation, and installed software versions
  are mutually compatible.
- **Automatic device selection:** The training and inference code can
  select CUDA when it is available and otherwise fall back to CPU.

GPU acceleration is optional; a dedicated GPU is not required to run
the project. Training speed and the batch size that can be used may
differ between CPU and GPU systems. The experiments documented in this
README were recorded using CPU execution.

Record the actual Python, PyTorch, Torchvision, NumPy, pandas, and
scikit-learn versions for each run.

The virtual environment is local to each machine and should not be
committed to GitHub.

### Activate the virtual environment

From the project root:

``` powershell
.\.venv\Scripts\Activate.ps1
```

Check Python:

``` powershell
python --version
```

If PowerShell blocks activation, use an appropriate execution-policy
setting for the current user or invoke the virtual-environment Python
executable directly.

## 7. Dataset and Preprocessing Verification

The dataset and preprocessing pipeline was tested before full training.

Verification covered:

-   Loading the split CSV files.
-   Checking image paths.
-   Confirming class coverage across splits.
-   Loading images and labels through PyTorch DataLoaders.
-   Checking image tensor shape `(3, 224, 224)`.
-   Checking image tensor dtype `torch.float32`.
-   Checking label tensor dtype `torch.int64`.
-   Checking tensor values for finite numbers.
-   Executing training and evaluation transformations.
-   Verifying model forward and backward passes.

The project records successful dataset and preprocessing checks. For any
new run, recheck paths because the CSVs may contain absolute paths from
another computer.

## 8. Baseline CNN Architecture and Training

The baseline is a custom PyTorch CNN named `BaselineCNN`.

### Architecture

The network contains four convolutional stages with output channels:

1.  32 channels
2.  64 channels
3.  128 channels
4.  256 channels

The architecture uses convolutional layers, batch normalization, ReLU
activations, and max pooling in the earlier stages. Adaptive average
pooling produces a fixed-size feature representation.

The classification head consists of:

-   Flattening
-   Dropout with probability 0.3
-   A fully connected layer producing 38 class scores

The project architecture record reports approximately **399,142
trainable parameters**. Verify this count against the exact source
file/checkpoint if it is quoted in a formal report.

### Training configuration

  Parameter           Value
  ------------------- ---------------
  Model               BaselineCNN
  Training images     115,150
  Validation images   24,675
  Recorded epochs     20
  Batch size          4
  Learning rate       0.001
  Optimizer           Adam
  Loss function       Cross-entropy
  Input resolution    224 x 224

### Complete recorded training history

The following values were supplied in the project README. Accuracy is
displayed as a percentage.

  -----------------------------------------------------------------------------
        Epoch  Train Loss        Train        Val Loss Val Accuracy    Duration
                              Accuracy                                      (s)
  ----------- ----------- ------------ --------------- ------------ -----------
            1   1.4561616       58.18%       0.7468221       78.75%     7431.21

            2   0.7605283       76.91%       0.4972142       85.59%     6644.92

            3   0.5638184       82.85%       0.3969425       88.58%     6407.29

            4   0.4580436       85.87%       0.3329189       90.04%     6516.69

            5   0.3939223       87.85%       0.2599354       92.31%     6501.87

            6   0.3539882       89.06%       0.2971917       90.97%     6458.10

            7   0.3193868       90.07%       0.2168564       93.25%     6444.67

            8   0.2922549       90.99%       0.1886616       94.43%     6524.96

            9   0.2714475       91.52%       0.2205243       93.17%     7240.53

           10   0.2572454       92.04%       0.2009383       93.81%     6666.68

           11   0.2410960       92.42%       0.1611719       95.08%     7620.03

           12   0.2315910       92.77%       0.1536519       95.43%     6574.57

           13   0.2203640       93.11%       0.1415694       95.79%     6403.88

           14   0.2092829       93.43%       0.1764299       94.38%     6398.92

           15   0.2035673       93.51%       0.1334466       95.99%     6389.38

           16   0.1944735       93.83%       0.1340617       95.91%     6420.04

           17   0.1875102       94.08%   **0.1273499**   **96.36%**     6568.27

           18   0.1863113       94.16%       0.1415347       95.68%     6529.21

           19   0.1790769       94.39%       0.1366558       95.81%     6523.25

           20   0.1725044   **94.54%**       0.1324786       96.02%     6410.56
  -----------------------------------------------------------------------------

Recorded observations:

-   Highest validation accuracy: **96.3607% at Epoch 17**.
-   Lowest validation loss: **0.1273499 at Epoch 17**.
-   Training accuracy at Epoch 20: approximately **94.54%**.
-   Training loss at Epoch 20: **0.1725044**.
-   Total recorded training time: approximately **36.9 hours**.

The checkpoint used by the hybrid experiment is reported as Epoch 17,
with best validation accuracy 0.9636068896. This agrees with the
supplied training-history best epoch.

### Checkpoint provenance and paths

The handoff records these paths:

``` text
outputs/baseline_cnn_full/best_model.pth
outputs/baseline_cnn_retrained/best_model.pth
```

The later hybrid run summary identifies this checkpoint:

``` text
outputs/baseline_cnn_retrained/best_model.pth
```

and records:

-   Epoch: 17
-   Number of classes: 38
-   Seed: 42
-   Best validation accuracy: 0.9636068895643364

The supplied README also associates the 20-epoch training history with
the `baseline_cnn_retrained` folder. Verify the actual files and
metadata before treating the two checkpoint paths as interchangeable.
Preserve both checkpoints and their histories.

The checkpoint structure was inspected and includes:

``` text
epoch
model_state_dict
optimizer_state_dict
best_val_accuracy
num_classes
class_names
seed
history
train_generator_state
torch_rng_state
numpy_rng_state
python_rng_state
```

## 10. Hybrid CNN Feature Extraction and SVM

The hybrid approach uses the trained CNN as a feature extractor and
LinearSVC as the downstream classifier.

### Feature extraction

Reported settings:

  Setting                            Value
  ---------------------------------- -------------------------
  Image size                         224 x 224
  Feature dimension                  256
  Batch size                         64
  Device in recorded run             CPU
  Normalization mean                 `[0.485, 0.456, 0.406]`
  Normalization standard deviation   `[0.229, 0.224, 0.225]`

### SVM configuration

``` python
LinearSVC(
    C=1.0,
    class_weight="balanced",
    random_state=42,
    max_iter=5000
)
```

The scaler and classifier are fitted using training features only.
Validation features are used for hyperparameter selection. Test features
are reserved for final evaluation.

### Feature files

The completed hybrid run is stored at:

``` text
outputs/hybrid_svm_run_20261029_125118/
```

Saved feature files:

``` text
train_features.npz
val_features.npz
test_features.npz
```

Reported feature dimensions:

  Split           Images Feature shape
  ------------ --------- -----------------
  Training       115,150 `(115150, 256)`
  Validation      24,675 `(24675, 256)`
  Test            24,676 `(24676, 256)`

### Initial hybrid metrics

  Metric              Validation       Test
  ----------------- ------------ ----------
  Accuracy              97.4914%   97.5361%
  Macro precision       96.5528%   96.2924%
  Macro recall          96.6293%   96.7466%
  Macro F1              96.5851%   96.5064%
  Weighted F1           97.4936%   97.5401%
  Samples                 24,675     24,676

The hybrid run summary records the baseline checkpoint, 38 classes,
Epoch 17, seed 42, feature dimension 256, batch size 64, and CPU device.

### Important hybrid artifacts

``` text
outputs/hybrid_svm_run_20261029_125118/
    hybrid_svm.joblib
    per_class_analysis.csv
    run_summary.json
    val_classification_report.csv
    val_confusion_matrix.csv
    val_metrics.json
    test_classification_report.csv
    test_confusion_matrix.csv
    test_metrics.json
    train_features.npz
    val_features.npz
    test_features.npz
```

The actual directory may contain additional files, including the tuning
subdirectory.

## 11. SVM Hyperparameter Comparison

Three values of the SVM regularization parameter `C` were compared using
validation performance.

  --------------------------------------------------------------------------------------
            C     Validation          Macro   Macro recall       Macro F1    Weighted F1
                    accuracy      precision                               
  ----------- -------------- -------------- -------------- -------------- --------------
          0.1       97.4022%       96.1607%       96.6288%       96.3868%       97.4032%

      **1.0**   **97.4914%**   **96.5528%**   **96.6293%**   **96.5851%**   **97.4936%**

         10.0       97.2847%       96.2671%       96.1687%       96.2031%       97.2871%
  --------------------------------------------------------------------------------------

The selected setting was `C = 1.0`, which achieved the highest
validation macro F1 and validation accuracy among the tested values.

Tuning reports were recorded under:

``` text
outputs/hybrid_svm_run_20261029_125118/svm_tuning_20261029_142949/
```

Reported files include:

``` text
validation_report_C_0.1.csv
validation_report_C_1.0.csv
validation_report_C_10.0.csv
```

The tuning process reused saved CNN features and did not retrain the
CNN.

## 12. Baseline CNN Validation Report

A baseline validation report was subsequently generated so that baseline
and hybrid models could be compared on the same validation split.

Validation split:

``` text
data/processed/splits/val.csv
```

Number of samples: **24,675**.

### Baseline validation metrics

  Metric                 Baseline CNN
  -------------------- --------------
  Accuracy                   96.3607%
  Macro precision              95.58%
  Macro recall                 94.67%
  Macro F1                     94.92%
  Weighted F1                  96.37%
  Validation samples           24,675

The baseline checkpoint metadata records:

-   Checkpoint epoch: 17
-   Best validation accuracy: 0.9636068895643364
-   Number of classes: 38
-   Seed: 42

### Validation comparison

  Metric              Baseline CNN   Hybrid CNN + SVM
  ----------------- -------------- ------------------
  Accuracy                96.3607%           97.4914%
  Macro precision           95.58%           96.5528%
  Macro recall              94.67%           96.6293%
  Macro F1                  94.92%           96.5851%
  Weighted F1               96.37%           97.4936%
  Samples                   24,675             24,675

Approximate hybrid improvements on validation:

-   Accuracy: **+1.13 percentage points**.
-   Macro F1: **+1.67 percentage points**.

### Validation structural alignment

The validation reports and confusion matrices were checked for:

-   38 classes in each report.
-   Identical class names.
-   Matching per-class support counts.
-   Matching confusion matrix labels.
-   Matching confusion matrix row totals.
-   24,675 total samples in each evaluation.

**Structural alignment: Passed.**

This confirms matching report structure, class names, and support
counts. It does not independently prove identical image bytes, image
ordering, or preprocessing. Those should be checked in the underlying
inference scripts before making a strict end-to-end equivalence claim.

## 13. Baseline and Hybrid Test Evaluation

The held-out test split contains **24,676 samples**.

### Baseline CNN test metrics

The baseline `test_metrics.json` records:

``` json
{
  "checkpoint_epoch": 17,
  "best_validation_accuracy": 0.9636068895643364,
  "test_accuracy": 0.96020424704166,
  "test_samples": 24676,
  "num_classes": 38
}
```

The baseline macro F1 was calculated from the per-class test report as
approximately **94.3328%**.

The baseline weighted F1 value of **96.0257%** appears in the supplied
project README comparison table. Verify it against the original baseline
test classification report before treating it as a confirmed metric.

### Hybrid CNN + SVM test metrics

The hybrid `test_metrics.json` records:

``` json
{
  "accuracy": 0.9753606743394392,
  "macro_precision": 0.9629242842635041,
  "macro_recall": 0.96746607203742,
  "macro_f1": 0.9650640723463869,
  "weighted_f1": 0.975400752452119,
  "samples": 24676,
  "split": "test"
}
```

### Test comparison

  -----------------------------------------------------------------------
  Metric                           Baseline CNN          Hybrid CNN + SVM
  ------------------- ------------------------- -------------------------
  Accuracy                             96.0204%                  97.5361%

  Macro F1                             94.3328%                  96.5064%

  Weighted F1           Verify report; supplied                  97.5401%
                           README says 96.0257% 

  Test samples                           24,676                    24,676
  -----------------------------------------------------------------------

Using the recorded accuracy and macro F1 values:

-   Accuracy improvement: approximately **1.52 percentage points**.
-   Macro F1 improvement: approximately **2.17 percentage points**.

The confusion matrix totals indicate:

  Model                Incorrect predictions
  ------------------ -----------------------
  Baseline CNN                           982
  Hybrid CNN + SVM                       608
  Difference                374 fewer errors

This is approximately 38.1% fewer incorrect predictions for the hybrid
model on the recorded test evaluation.

The comparison reports matching 38 class names and support counts.
Before making a final scientific claim, retain the preprocessing and
checkpoint provenance checks described above.

## 14. Class-wise Test Performance

The per-class reports were compared to identify which classes benefited
most from the hybrid model.

### Largest reported F1 improvements

  Class                                  Hybrid F1 change
  ---------------------------- --------------------------
  Potato_healthy                 +16.55 percentage points
  Tomato_Early_blight             +5.68 percentage points
  Tomato_Tomato_mosaic_virus      +5.58 percentage points
  Corn_Northern_Leaf_Blight       +5.44 percentage points
  Tomato_Late_blight              +4.82 percentage points
  Tomato_Target_Spot              +4.06 percentage points
  Potato_Late_blight              +3.28 percentage points
  Peach_healthy                   +3.13 percentage points
  Tomato_Leaf_Mold                +2.87 percentage points
  Corn_Gray_leaf_spot             +2.78 percentage points

### Notable regressions

  Class                             Hybrid F1 change
  ------------------------ -------------------------
  Apple_Cedar_apple_rust     -1.16 percentage points
  Pepper_Bacterial_spot      -0.74 percentage points

The hybrid improves overall performance, but individual classes may show
small regressions.

## 15. Confusion Matrix and Misclassification Analysis

The test confusion matrices show strong diagonal patterns, indicating
that most samples are correctly classified. Some visually similar
disease categories remain difficult.

### Corn disease pair

Classes:

``` text
Corn_Gray_leaf_spot
Corn_Northern_Leaf_Blight
```

  Actual → Predicted                                  Baseline   Hybrid
  ------------------------------------------------- ---------- --------
  Corn_Northern_Leaf_Blight → Corn_Gray_leaf_spot           63       24
  Corn_Gray_leaf_spot → Corn_Northern_Leaf_Blight           24       36
  **Combined directional confusion**                    **87**   **60**

Combined confusion decreased by 27 cases.

### Tomato disease pair

Classes:

``` text
Tomato_Early_blight
Tomato_Late_blight
```

  Actual → Predicted                           Baseline   Hybrid
  ------------------------------------------ ---------- --------
  Tomato_Early_blight → Tomato_Late_blight           44       13
  Tomato_Late_blight → Tomato_Early_blight           14       27
  **Combined directional confusion**             **58**   **40**

Combined confusion decreased by 18 cases.

For both pairs, one direction improved substantially while the reverse
direction worsened. Reporting both directions is necessary; a one-way
improvement alone does not show that both classes became easier to
distinguish.

### Other major improvements

  Actual class                Incorrect prediction      Baseline   Hybrid
  --------------------------- ----------------------- ---------- --------
  Potato_healthy              Potato_Late_blight              24        5
  Potato_Early_blight         Potato_Late_blight              24        8
  Peach_Bacterial_spot        Peach_healthy                   17        2
  Tomato_Bacterial_spot       Tomato_Late_blight              15        3
  Apple_healthy               Soybean_healthy                 11        0
  Tomato_Septoria_leaf_spot   Tomato_Bacterial_spot           19        8
  Apple_Scab                  Apple_healthy                   10        1
  Grape_Esca                  Grape_Black_rot                 15        6

### Notable regressions in directional confusions

  Actual class                Incorrect prediction          Baseline   Hybrid
  --------------------------- --------------------------- ---------- --------
  Tomato_Late_blight          Tomato_Early_blight                 14       27
  Corn_Gray_leaf_spot         Corn_Northern_Leaf_Blight           24       36
  Apple_Scab                  Apple_Cedar_apple_rust               3        7
  Tomato_Septoria_leaf_spot   Apple_Scab                           1        5
  Potato_Late_blight          Potato_healthy                       0        4

Use validation results to decide what to improve in future experiments.
Do not use the test error list to tune the model.

## 16. Saved Validation Feature Compatibility Investigation

The hybrid experiment already saved:

``` text
outputs/hybrid_svm_run_20261029_125118/val_features.npz
```

The saved features have a reported dimension of 256.

A possible faster method for generating baseline CNN validation
predictions is to apply the baseline CNN's original classification head
directly to these saved feature vectors.

This is only valid if the following are verified:

1.  The 256-dimensional features are taken from exactly the same layer
    expected by the original CNN classification head.
2.  The feature extraction implementation matches the baseline
    architecture.
3.  The saved features use the same preprocessing as baseline
    evaluation.
4.  The saved labels and sample order correspond to the same validation
    CSV rows.
5.  The original classifier head is reconstructed and loaded with the
    correct checkpoint weights.

Do not assume compatibility solely because both feature vectors have 256
elements.

### Inspect the relevant files

``` powershell
Get-Content .\baseline_cnn.py
```

``` powershell
Get-Content .\scripts\run_hybrid_svm.py
```

Inspect saved feature arrays:

``` powershell
@'
import numpy as np

path = "outputs/hybrid_svm_run_20261029_125118/val_features.npz"
data = np.load(path)

print("Arrays:", data.files)

for key in data.files:
    arr = data[key]
    print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
    if arr.size:
        print("First values:", arr.reshape(-1)[:5])
'@ | python -
```

If compatibility is proven, the baseline predictions may be
reconstructed from the saved features without reloading all validation
images. This would be an inference-only procedure, not retraining.

## 17. Important Output Directories

### Baseline outputs

``` text
outputs/
├── baseline_cnn/
├── baseline_cnn_full/
└── baseline_cnn_retrained/
```

The handoff identifies the hybrid checkpoint as:

``` text
outputs/baseline_cnn_retrained/best_model.pth
```

Baseline evaluation artifacts include:

``` text
test_classification_report.csv
test_confusion_matrix.csv
test_metrics.json
training_history.csv
```

The baseline validation evaluation also generated validation report,
confusion matrix, and metrics files. Confirm their exact folder and
filenames in the current checkout before scripting against them.

### Hybrid run

``` text
outputs/hybrid_svm_run_20261029_125118/
```

Important files:

``` text
hybrid_svm.joblib
per_class_analysis.csv
run_summary.json
val_classification_report.csv
val_confusion_matrix.csv
val_metrics.json
test_classification_report.csv
test_confusion_matrix.csv
test_metrics.json
train_features.npz
val_features.npz
test_features.npz
```

### Test comparison output

A separate comparison directory was generated:

``` text
outputs/test_evaluation_comparison_20261101_143406/
```

It contains:

``` text
test_metrics_comparison.csv
test_per_class_comparison.csv
baseline_test_confusion_matrix.csv
hybrid_test_confusion_matrix.csv
baseline_test_normalized_confusion_matrix.csv
hybrid_test_normalized_confusion_matrix.csv
comparison_summary.json
test_metrics_comparison.png
test_per_class_f1_comparison.png
test_per_class_f1_change.png
baseline_test_normalized_confusion_matrix.png
hybrid_test_normalized_confusion_matrix.png
test_misclassification_analysis.csv
```

This directory was created separately so that existing model and
evaluation artifacts remained intact.

## 18. Important Python Scripts

The project includes the following files:

``` text
baseline_cnn.py
check_model.py
check_torch.py
continue_baseline.py
dataset_loader.py
evaluate_baseline.py
show_epoch_results.py
test_preprocessing.py
train_baseline.py
train_baseline_full.py
train_baseline_smoke.py
```

Scripts directory:

``` text
scripts/
├── analyze_hybrid_results.py
├── analyze_validation_per_class.py
├── build_hybrid_dataset.py
├── build_plantdoc_manifest.py
├── extract_plantdoc.py
├── inspect_dataset.py
├── run_hybrid_svm.py
└── tune_hybrid_svm.py
```

Model and data modules:

``` text
src/
├── data/
│   ├── clean_dataset.py
│   └── prepare_splits.py
└── models/
    ├── cnn_backbones.py
    ├── extract_deep_features.py
    └── train_cnn.py
```

The repository may contain additional scripts and directories. This is a
list of files reported in the handoff, not a guarantee that every file
is present in every checkout.

## 19. Useful PowerShell Commands

Open the project:

``` powershell
cd "C:\Users\ottaa\OneDrive\Desktop\Minor Project\plant-disease-hybrid"
```

Activate the virtual environment:

``` powershell
.\.venv\Scripts\Activate.ps1
```

List output directories:

``` powershell
Get-ChildItem .\outputs -Directory |
Select-Object Name, FullName
```

Find validation reports:

``` powershell
Get-ChildItem .\outputs -Recurse -File |
Where-Object {
    $_.Name -match "val.*(report|confusion)|validation.*(report|confusion)"
} |
ForEach-Object {
    $_.FullName
}
```

Inspect the hybrid run summary:

``` powershell
Get-Content .\outputs\hybrid_svm_run_20261029_125118\run_summary.json
```

Inspect baseline test metrics:

``` powershell
Get-Content .\outputs\baseline_cnn_retrained\test_metrics.json
```

Inspect hybrid test metrics:

``` powershell
Get-Content .\outputs\hybrid_svm_run_20261029_125118\test_metrics.json
```

## 20. Reproducibility, GitHub, and Artifact Preservation

Avoid committing large datasets, virtual environments, and model
binaries unless explicitly required by the project.

A recommended `.gitignore` baseline is:

``` gitignore
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

Adding a pattern to `.gitignore` does not remove files already tracked
by Git.

For reproducibility, record:

-   Python, PyTorch, Torchvision, NumPy, pandas, and scikit-learn
    versions.
-   Processing device used for each run (CPU or GPU/CUDA).
-   Dataset version and split definitions.
-   Model architecture and checkpoint identity.
-   Training configuration and random seed.
-   Training history and checkpoint-selection criterion.
-   SVM configuration and selected hyperparameters.
-   Final test metrics and class-level reports.
-   Image preprocessing and normalization configuration.

### Experiment preservation rules

1.  Do not retrain the baseline CNN unless explicitly requested.
2.  Do not retrain the SVM unless explicitly requested.
3.  Do not overwrite existing checkpoints.
4.  Do not overwrite existing reports, confusion matrices, or feature
    arrays.
5.  Save new experiments in separate timestamped directories.
6.  Use validation data for model comparison and hyperparameter
    selection.
7.  Do not use test data for model selection.
8.  Preserve the original train, validation, and test splits.
9.  Preserve the 38-class mapping and ordering.
10. Verify image paths before loading images.
11. Record whether each experiment used CPU or GPU/CUDA processing.

## 21. Next Development Milestones

### Milestone 1: Verify checkpoint provenance

-   Confirm which checkpoint corresponds to the supplied 20-epoch
    history.
-   Verify the checkpoint-selection criterion and selected epoch.
-   Preserve original model weights and logs.

### Milestone 2: Finish reproducibility checks

-   Confirm the baseline and hybrid use the same validation/test
    examples.
-   Verify label ordering and image preprocessing.
-   Confirm the baseline weighted F1 from its original classification
    report.
-   Verify the source and metadata of each checkpoint.

### Milestone 3: Verify saved-feature compatibility

-   Inspect `baseline_cnn.py`.
-   Inspect `scripts/run_hybrid_svm.py`.
-   Inspect `val_features.npz`.
-   Confirm whether the baseline classification head can consume the
    saved feature vectors.
-   Do not assume compatibility without code-level verification.

### Milestone 5: Inference API

-   Implement model loading and inference.
-   Integrate the exact training/evaluation preprocessing.
-   Return the predicted class and appropriately described confidence
    information.
-   Test inference using representative images.

### Milestone 6: Frontend and deployment

Frontend integration and deployment are not planned yet. They can be
considered after the inference API is stable and verified.

## 22. Final Project Summary

The project has completed dataset preparation, 38-class mapping,
stratified splitting, dataset/DataLoader verification, baseline CNN
training for 20 recorded epochs, CNN feature extraction, SVM tuning, and
initial hybrid evaluation.

The baseline CNN's best recorded validation accuracy is **96.3607% at
Epoch 17**. The hybrid CNN--SVM model with `LinearSVC(C=1.0)` achieved
**97.4914% validation accuracy** and **97.5361% test accuracy** in the
recorded experiment.

The baseline test evaluation records **96.0204% accuracy** and
approximately **94.3328% macro F1**, while the hybrid test evaluation
records **97.5361% accuracy** and **96.5064% macro F1**. The recorded
comparison indicates approximately **1.52 percentage points higher test
accuracy** and **2.17 percentage points higher test macro F1** for the
hybrid model.

These comparisons should be presented with the accompanying checkpoint,
preprocessing, split, and class-order verification. The recorded
experiments used CPU execution. The project itself supports CPU execution
and can use GPU acceleration with CUDA when a compatible GPU and
software environment are available.

**Current milestone:** Dataset preparation, baseline CNN training, and
initial hybrid CNN--SVM evaluation completed. The recorded experiments
used CPU processing; GPU/CUDA execution is an available option on
compatible systems.

**Next priorities:** Verify checkpoint and preprocessing provenance,
complete saved-feature compatibility checks, investigate remaining class
confusions using validation data, and implement the inference API when
planned.
