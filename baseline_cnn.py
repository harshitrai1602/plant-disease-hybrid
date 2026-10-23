
import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """Baseline CNN for 38-class plant disease classification."""

    def __init__(self, num_classes=38):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = BaselineCNN(num_classes=38).to(device)
    sample = torch.randn(2, 3, 224, 224, device=device)

    with torch.no_grad():
        output = model(sample)

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Output shape:", tuple(output.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Baseline CNN test passed.")
