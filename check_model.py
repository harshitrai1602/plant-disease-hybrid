print("CHECK_MODEL SCRIPT STARTED")

import torch
import torch.nn as nn
from torchvision import models

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))

model = models.mobilenet_v3_small(
    weights=None,
    num_classes=38
).to(device)

images = torch.randn(2, 3, 224, 224, device=device)
labels = torch.randint(0, 38, (2,), device=device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()

torch.cuda.synchronize()

print("Output shape:", tuple(outputs.shape))
print("Loss:", round(loss.item(), 4))
print("GPU memory allocated (MB):",
      round(torch.cuda.memory_allocated() / (1024 ** 2), 2))
print("SUCCESS: Forward and backward passes completed.")
