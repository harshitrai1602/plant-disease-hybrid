
from pathlib import Path
import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
SPLIT_DIR = PROJECT_DIR / "data" / "processed" / "splits"

TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV = SPLIT_DIR / "val.csv"
TEST_CSV = SPLIT_DIR / "test.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # Start with 0 on Windows

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# --------------------------------------------------
# 2. Image transformations
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# --------------------------------------------------
# 3. Resolve image paths
# --------------------------------------------------

def resolve_image_path(raw_path):
    """Use the recorded path, or resolve it under this repo."""

    raw_path = str(raw_path).strip().strip('"')
    original = Path(raw_path)

    if original.is_file():
        return original

    # The CSV may contain paths from the original D:\MAJOR
    # checkout. Reuse the path relative to this repository.
    normalized = raw_path.replace("/", "\\")
    marker = "plant-disease-hybrid\\"

    if marker.lower() in normalized.lower():
        index = normalized.lower().rfind(marker.lower())
        relative_part = normalized[index + len(marker):]
        candidate = PROJECT_DIR / Path(relative_part)

        if candidate.is_file():
            return candidate

    return original

# --------------------------------------------------
# 4. Custom CSV Dataset
# --------------------------------------------------

class PlantDiseaseDataset(Dataset):

    def __init__(self, csv_path, transform=None):
        self.csv_path = Path(csv_path)
        self.transform = transform

        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"CSV not found: {self.csv_path}"
            )

        self.data = pd.read_csv(self.csv_path)

        required = {
            "image_path", "label", "class_id"
        }

        missing = required - set(self.data.columns)

        if missing:
            raise ValueError(
                f"Missing CSV columns: {sorted(missing)}"
            )

        self.data = self.data.dropna(
            subset=["image_path", "label", "class_id"]
        ).reset_index(drop=True)

        self.data["class_id"] = pd.to_numeric(
            self.data["class_id"], errors="raise"
        )

        if not (self.data["class_id"] % 1 == 0).all():
            raise ValueError("class_id must contain integers.")

        self.data["class_id"] = self.data["class_id"].astype(int)

        if (self.data["class_id"] < 0).any():
            raise ValueError("class_id cannot be negative.")

        # Resolve paths without loading all image pixels.
        self.paths = [
            resolve_image_path(p)
            for p in self.data["image_path"]
        ]

        self.labels = self.data["class_id"].tolist()
        self.class_names = (
            self.data.groupby("class_id")["label"]
            .agg(lambda s: s.mode().iloc[0])
            .to_dict()
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.paths[index]
        label = self.labels[index]

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                if self.transform:
                    image = self.transform(image)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Could not read image at row {index}: "
                f"{image_path}"
            ) from exc

        return image, torch.tensor(label, dtype=torch.long)

# --------------------------------------------------
# 5. Build datasets and loaders
# --------------------------------------------------

train_dataset = PlantDiseaseDataset(
    TRAIN_CSV, transform=train_transform
)
val_dataset = PlantDiseaseDataset(
    VAL_CSV, transform=eval_transform
)
test_dataset = PlantDiseaseDataset(
    TEST_CSV, transform=eval_transform
)

# Check class IDs have consistent meanings across splits.
def label_map(dataset):
    return dataset.data.groupby("class_id")["label"].agg(
        lambda s: set(s.astype(str))
    ).to_dict()

train_map = label_map(train_dataset)

for name, dataset in [
    ("validation", val_dataset),
    ("test", test_dataset),
]:
    for class_id, names in label_map(dataset).items():
        if class_id not in train_map:
            raise ValueError(
                f"{name} contains class_id {class_id}, "
                "which is missing from training data."
            )
        if not names.issubset(train_map[class_id]):
            raise ValueError(
                f"Inconsistent label for class_id {class_id}"
            )

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)

# --------------------------------------------------
# 6. Verify paths, classes and batches
# --------------------------------------------------

def main():
    print("Project:", PROJECT_DIR)
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    for name, dataset in [
        ("Train", train_dataset),
        ("Validation", val_dataset),
        ("Test", test_dataset),
    ]:
        missing_paths = [
            str(path) for path in dataset.paths
            if not path.is_file()
        ]

        print(f"\n{name} images in CSV: {len(dataset)}")
        print(f"{name} missing image files: {len(missing_paths)}")
        print(
            f"{name} unique classes:",
            dataset.data["class_id"].nunique()
        )

        if missing_paths:
            print("First missing path:", missing_paths[0])

    if any(
        not path.is_file()
        for path in train_dataset.paths
    ):
        raise FileNotFoundError(
            "Some training images are missing. "
            "Check the original image directory and CSV paths."
        )

    # Load a sample batch.
    images, labels = next(iter(train_loader))

    print("\nBatch verification")
    print("Images:", tuple(images.shape))
    print("Labels:", tuple(labels.shape))
    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)

    images = images.to(DEVICE, non_blocking=True)
    labels = labels.to(DEVICE, non_blocking=True)

    print("Image device:", images.device)
    print("Label device:", labels.device)
    print("Sample labels:", labels[:10].tolist())

    print("\nDataset and DataLoader verification passed!")

if __name__ == "__main__":
    main()
