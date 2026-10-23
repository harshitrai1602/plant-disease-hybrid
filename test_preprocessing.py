import torch
import torchvision
from PIL import Image
from torchvision import transforms

print("=== Plant Disease Preprocessing Test ===")
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)

IMAGE_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Training transformations: random augmentation enabled
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.02
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])

# Validation/test transformations: no random augmentation
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])

# Create a sample RGB image for testing
sample_image = Image.new("RGB", (400, 300), color="green")

train_tensor = train_transform(sample_image)
eval_tensor = eval_transform(sample_image)

print("\n--- Training Pipeline ---")
print("Tensor shape:", train_tensor.shape)
print("Data type:", train_tensor.dtype)
print("All values finite:", torch.isfinite(train_tensor).all().item())

print("\n--- Evaluation Pipeline ---")
print("Tensor shape:", eval_tensor.shape)
print("Data type:", eval_tensor.dtype)
print("All values finite:", torch.isfinite(eval_tensor).all().item())

# Verify expected outputs
assert train_tensor.shape == (3, 224, 224)
assert eval_tensor.shape == (3, 224, 224)
assert torch.isfinite(train_tensor).all()
assert torch.isfinite(eval_tensor).all()
assert train_tensor.dtype == torch.float32
assert eval_tensor.dtype == torch.float32

print("\nSUCCESS: Preprocessing and augmentation tests passed!")
